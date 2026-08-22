import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model_spec import build_transformer
import train

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

def create_kv_causal_mask(q_len, kv_past_len, device):
    """
    Builds a causal mask compatible with KV Caching.
    q_len: number of incoming query tokens
    kv_past_len: number of cached key/value tokens already stored
    """
    if q_len == 1:
        # Single token decoding: can attend to all past keys + itself
        return torch.ones((1, 1, 1, kv_past_len + 1), dtype=torch.bool, device=device)
    
    # Multi-token queries (verification pass)
    c_mask = causal_mask(q_len).to(device)
    if c_mask.dim() == 3:
        c_mask = c_mask.unsqueeze(1)
        
    past_mask = torch.ones((1, 1, q_len, kv_past_len), dtype=torch.bool, device=device)
    return torch.cat([past_mask, c_mask], dim=-1)


def truncate_cache(kv_cache, keep_length):
    if kv_cache is None:
        return None

    truncated_cache = []
    for layer_cache in kv_cache:
        past_key, past_value = layer_cache
        past_key = past_key[:, :, :keep_length, :]
        past_value = past_value[:, :, :keep_length, :]
        truncated_cache.append((past_key, past_value))

    return truncated_cache


def spec_decode(model_p, model_q, source, source_mask, tokenizer_tgt, max_len, device, gamma=4):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    with torch.no_grad():
        encoder_output_p = model_p.encode(source, source_mask)
        encoder_output_q = model_q.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Initial Prefill: Process [SOS] token
    with torch.no_grad():
        init_mask = create_kv_causal_mask(1, 0, device)
        out_p, kv_cache_p = model_p.decode(encoder_output_p, source_mask, decoder_input, init_mask, kv_cache=None)
        out_q, kv_cache_q = model_q.decode(encoder_output_q, source_mask, decoder_input, init_mask, kv_cache=None)

        prob_p = model_p.project(out_p[:, -1])
        first_token = torch.argmax(prob_p, dim=-1, keepdim=True)
        decoder_input = torch.cat([decoder_input, first_token], dim=1)

    while decoder_input.size(1) < max_len:
        if decoder_input[0, -1].item() == eos_idx:
            break

        # -------------------------------------------------------------
        # STEP 1: Draft Phase (Model Q generates gamma tokens)
        # -------------------------------------------------------------
        draft_tokens = []
        current_draft_input = decoder_input[:, -1:]

        with torch.no_grad():
            for _ in range(gamma):
                kv_past_q = kv_cache_q[0][0].size(2) if kv_cache_q is not None else 0
                draft_mask = create_kv_causal_mask(1, kv_past_q, device)

                out_q, kv_cache_q = model_q.decode(
                    encoder_output_q, source_mask, current_draft_input, draft_mask, kv_cache=kv_cache_q
                )
                prob_q = model_q.project(out_q[:, -1])
                next_word_q = torch.argmax(prob_q, dim=-1, keepdim=True)

                draft_tokens.append(next_word_q)
                current_draft_input = next_word_q

        draft_tensor = torch.cat(draft_tokens, dim=1) # Shape: (1, gamma)

        # -------------------------------------------------------------
        # STEP 2: Verification Phase (Model P evaluates draft in 1 pass)
        # -------------------------------------------------------------
        # Feed last accepted token + gamma draft tokens
        verify_input = torch.cat([decoder_input[:, -1:], draft_tensor], dim=1) # Shape: (1, gamma + 1)

        with torch.no_grad():
            kv_past_p = kv_cache_p[0][0].size(2) if kv_cache_p is not None else 0
            verify_mask = create_kv_causal_mask(verify_input.size(1), kv_past_p, device)

            out_p, kv_cache_p = model_p.decode(
                encoder_output_p, source_mask, verify_input, verify_mask, kv_cache=kv_cache_p
            )
            prob_p = model_p.project(out_p)
            target_tokens = torch.argmax(prob_p, dim=-1) # Shape: (1, gamma + 1)

        # -------------------------------------------------------------
        # STEP 3: Acceptance & Correction Logic
        # -------------------------------------------------------------
        accepted = 0
        for i in range(gamma):
            if draft_tensor[0, i] == target_tokens[0, i]:
                accepted += 1
            else:
                break

        # -------------------------------------------------------------
        # STEP 4: Sequence Update & KV Cache Maintenance
        # -------------------------------------------------------------
        if accepted == gamma:
            # All gamma tokens accepted + grab bonus token generated by M_p at index gamma
            bonus_token = target_tokens[:, gamma:gamma+1]
            accepted_chunk = torch.cat([draft_tensor, bonus_token], dim=1)
            decoder_input = torch.cat([decoder_input, accepted_chunk], dim=1)

            # M_p cache has all keys stored. Update M_q cache with bonus token to keep in sync.
            with torch.no_grad():
                kv_past_q = kv_cache_q[0][0].size(2) if kv_cache_q is not None else 0
                q_bonus_mask = create_kv_causal_mask(1, kv_past_q, device)
                _, kv_cache_q = model_q.decode(
                    encoder_output_q, source_mask, bonus_token, q_bonus_mask, kv_cache=kv_cache_q
                )
        else:
            # Append accepted tokens + correction token from target model
            correction_token = target_tokens[:, accepted:accepted+1]
            if accepted > 0:
                accepted_chunk = torch.cat([draft_tensor[:, :accepted], correction_token], dim=1)
            else:
                accepted_chunk = correction_token

            decoder_input = torch.cat([decoder_input, accepted_chunk], dim=1)

            # Truncate caches: cache length must equal decoder_input.size(1) - 1
            target_cache_len = decoder_input.size(1) - 1
            kv_cache_p = truncate_cache(kv_cache_p, target_cache_len)
            kv_cache_q = truncate_cache(kv_cache_q, target_cache_len)

    return decoder_input


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    model_q = model # Note: Replace with actual smaller draft model once instantiated

    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation set"

            model_out = spec_decode(
                model_p=model,
                model_q=model_q,
                source=encoder_input,
                source_mask=encoder_mask,
                tokenizer_tgt=tokenizer_tgt,
                max_len=max_len,
                device=device
            )

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out[0].detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f'SOURCE:    {source_text}')
            print_msg(f'TARGET:    {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]","[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    #Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max sentence length of source: {max_len_src}")
    print(f"Max sentence length of target: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len, d_model, n_layers, heads):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model, n_layers, heads)
    return model

def train_spec_model(config):
    #Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model_p = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['d_model'], n_layers=config['N'], heads=config['h']).to(device)
    model_q = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['q_d_model'], n_layers=config['q_N'], heads=config['q_h']).to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(list(model_p.parameters()) + list(model_q.parameters()), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        model_p.load_state_dict(state['model_p_state_dict'])
        model_q.load_state_dict(state['model_q_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model_p.train()
            model_q.train()

            encoder_input = batch['encoder_input'].to(device) #(B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device) #(B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device) #(B, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device) #(B, 1, Seq_Len, Seq_Len)
            label = batch['label'].to(device) #(B, Seq_Len)

            #Run tensors through transformer
            ##Model P (TARGET MODEL)
            encoder_output_p = model_p.encode(encoder_input, encoder_mask) #(B, Seq_Len, d_model)
            decoder_output_p, _ = model_p.decode(encoder_output_p, encoder_mask, decoder_input, decoder_mask) #(B, Seq_Len, d_model)
            proj_output_p = model_p.project(decoder_output_p) #(B, Seq_Len, tgt_vocab_size)
            #(B, Seq_Len, tgt_vocab_size) --> (B * Seq_Len, tgt_vocab_size)
            loss_p = loss_fn(proj_output_p.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            ##Model Q (DRAFT MODEL/APPROX MODEL)
            encoder_output_q = model_q.encode(encoder_input, encoder_mask)
            decoder_output_q, _ = model_q.decode(encoder_output_q, encoder_mask, decoder_input, decoder_mask)
            proj_output_q = model_q.project(decoder_output_q)
            loss_q = loss_fn(proj_output_q.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            loss = loss_p + loss_q

            batch_iterator.set_postfix({f"loss_p": f"{loss_p.item():6.3f}", f"loss_q": f"{loss_q.item():6.3f}"})

            #Log the loss
            writer.add_scalar('train loss p', loss_p.item(), global_step)
            writer.add_scalar('train loss q', loss_q.item(), global_step)
            writer.flush()

            #Backpropogate the loss
            optimizer.zero_grad()
            loss.backward()

            #Update the weights
            optimizer.step()

            global_step += 1

        run_validation(model_p, model_q, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        #Save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_q_state_dict': model_p.state_dict(),
            'model_q_state_dict': model_q.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_spec_model(config)