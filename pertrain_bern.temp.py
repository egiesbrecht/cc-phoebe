# %%
import glob
import time
import os
import math
from datasets import DatasetDict, load_dataset, Dataset, concatenate_datasets, load_from_disk
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch import nn
import transformers
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    BertTokenizer, 
    RobertaTokenizer,
    XLNetTokenizer,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np

from load_set import load_set, load_moses_set
from epoch_stats import EpochStats, print_stat_tuples
import model_training
from model_training import (
    BatchBuffer, 
    mask_tokens, 
    train_bern_model, 
    preprocess_for_maskedlm, 
    preprocess_for_causallm, 
    preprocess_for_monologe, 
    preprocess_for_sparselm, 
    preprocess_for_binary_sparselm,
    num_parameters, 
    num_trainable_parameters
)
import bert_modeling as bert
import bern_i2C_modeling as bern_i2C
import bern_i2_modeling as bern_i2
import bert_i1_modeling as bert_i1
import bert_i1_1_modeling as bert_i1_1
import bert_i1_2_modeling as bert_i1_2
import bert_i1_3_modeling as bert_i1_3
import bern_i3_modeling as bern_i3
import bern_i3A_modeling as i3a
import bern_i3B_modeling as i3b
import rffn_modeling as rffn
import rann_modeling as rann
import srm_modeling as srm
import lrm_modeling as lrm
import retentive_routing as rr
import rr_blocks_modeling as rrb
#import rrb_backups_01 as rrb
import smpl_rrb_modeling as smpl_rrb
import fold_mdim_rb_modeling as mdrb
import rbsrm_modeling as rbsrm
import stacked_rb_modeling as srb
import bounce_modeling as bounce
import coin_i1_modeling as ci1
import coin_i2A_modeling as ci2A
import coin_i2B_modeling as ci2B
import coin_i2C_modeling as ci2C

# %%
REBATCH = False
BATCH_SIZE = 8
BASE_PATH = "tmp_models/COIN-i2C_oasst1_25k_1x6_0dec-none_parallel_no-decay/"
#BASE_PATH = "tmp_models/COIN-i2B_mcca-en_0029_2x2_00dec_none-decoder-out_no-gate_parallel/"
#BASE_PATH = "tmp_models/RRB_oasst1_35k_2-2-encoder_0-1-decoder_decay_maskedLM.15_.2share_docx1/"

#BASE_PATH = "tmp_models/COIN-i2A_oasst1_25k_4reg_1-schema_decay_sparse-grouped-maskedLM.15_.2share_docx1/"
#BASE_PATH = "tmp_models/backup-RRB_oasst1_25k_2-2-encoder_0-1-decoder_decay_maskedLM.15_.2share_docx1_control-unit-R/"
if REBATCH:
    if BASE_PATH[-1] == "/":
        BASE_PATH = BASE_PATH[:-1]
    BASE_PATH += "_rebatch/"
DATASET_JSON_PATH = f"{BASE_PATH}/dataset/"
CHECKPOINT_PATH = f"{BASE_PATH}/model/"
WORDPIECE_TOKENIZER_DIR = f"{BASE_PATH}/wordpiece_tokenizer/"
BPE_TOKENIZER_DIR = f"{BASE_PATH}/bpe_tokenizer/"
SENTENCE_PIECE_TOKENIZER_DIR = f"{BASE_PATH}/sentence_piece_tokenizer/"
USE_CUSTOM_DATALOADER = False
LEARNING_RATE = 1e-5
EPS = 1e-8
EPOCHS = 10

# %%
SHUFFLE_TRAIN_DATA = False
SHUFFLE_TEST_DATA = False

# %%
CAUSAL_LM = False
ENCODE_CAUSAL_LM = False
GROUP_TEXTS = True
SPARSIFY = True
MASK_TOKEN = None
PAD_TOKEN = None
PREFIX = None
#PREFIX = "Replace all of the mask-tokens: "
#PREFIX = "This sentence is completely obsolete "
SWITCH_II_DECODER_II = False

# %%
f"out dir: {BASE_PATH}"

# %%
VOCAB_SIZE = 30_522
#VOCAB_SIZE = 32_000
#VOCAB_SIZE = 52_000
MAX_POSITION_EMBEDDINGS = 512
#MAX_POSITION_EMBEDDINGS = 516
#MAX_POSITION_EMBEDDINGS = 768
IS_HF_MODEL = False
IS_ENCODER_DECODER_MODEL = False
EPOCH_I = 0

# %%
if 0:
    model = rbsrm.RBSRMForMaskedLM(
        config=rbsrm.RBSRMConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,

            apply_decay=True,
            num_decay_parts=1,
            forward_method="parallel",
            
            block_schema=[4, 4],
            decoder_schema=[0, 1],
            K_schema=[2, 2],
            #apply_block_wise_ffn=True,

            hidden_retention_act="relu",
            hidden_out_act="relu",
        )
    )

# %%
if 0:
    model = smpl_rrb.RRBForMaskedLM(
        config=smpl_rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,

            apply_decay=True,
            num_decay_parts=1,
            #apply_attention_mask=True
            forward_method="parallel",
            copy_first_decoder_ids=True,

            block_schema=[[2], [4], [4], [2]],
            block_decoder_schema=[0, 0, 1, 1],
            #layer_decoder_schema=[0, 1],
            #apply_hierarchical_ffn=True,
            collector_method=4,
            extend_X=True,
            sum_in_qkv=False,
            qkv_sum_slice=False,
        )
    )

# %%
if 0:
    model = mdrb.MDRBForMaskedLM(
        config=mdrb.MDRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,

            apply_decay=True,
            num_decay_parts=1,
        )
    )

# %%
if 0:
    EPOCH_I = 8
    tokenizer = BertTokenizer.from_pretrained(f"{BASE_PATH}/wordpiece_tokenizer/")
    model = rrb.RRBForMaskedLM.from_pretrained(f"{BASE_PATH}/model/epoch_{EPOCH_I-1}/model")

# %%
if 0:
    model = srb.RRBForMaskedLM(
        config=srb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,

            apply_decay=True,
            num_decay_parts=1,
            #apply_attention_mask=True
            forward_method="parallel",
            copy_first_decoder_ids=True,
            decoder_pipeline=False,
            apply_dense_gate=False,

            block_schema=[2],
            #layer_decoder_schema=[0, 1],
            block_decoder_schema=[1],
            #apply_layer_wise_ffn=False,
            #apply_block_wise_ffn=True,
            #ffn_intermediate_factor=2,
            hidden_out_act="relu",
            stack_QKV=False,
        )
    )

# %%
if 0:
    model = rrb.RRBForMaskedLM(
        config=rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,

            apply_decay=True,
            num_decay_parts=1,
            #apply_attention_mask=True
            forward_method="parallel",
            copy_first_decoder_ids=True,
            decoder_pipeline=False,
            apply_dense_gate=False,

            block_schema=[2, 2],
            #layer_decoder_schema=[0, 1],
            block_decoder_schema=[0, 1],
            #apply_layer_wise_ffn=False,
            #apply_block_wise_ffn=True,
            #ffn_intermediate_factor=2,
            num_repetitions=1,
            apply_decoder_heads=False,

            #hidden_retention_act=None,
            #hidden_out_act=None,
        )
    )

# %%
if 0:
    model = ci2A.COINForMaskedLM(
        config=ci2A.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            #num_labels=NUM_LABELS,
        ),
        regions=[
            ci2A.COINRegionConfig(
                vocab_size=VOCAB_SIZE,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                apply_decay=True,
                num_decay_parts=1,
                forward_method="parallel",
                #num_layers=1,
                #is_decoder=True,
                #hidden_retention_act=None,
                #hidden_out_act=None,
                reverse_decay=True,
                decoder_schema=[1],
            )
        ] * 4,
    )
    """
        encoder=ci2A.COINRegionConfig(
                vocab_size=VOCAB_SIZE,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                apply_decay=True,
                num_decay_parts=1,
                forward_method="parallel",
                num_layers=1,
                is_decoder=False,
                #hidden_retention_act=None,
                #hidden_out_act=None,
                reverse_decay=True,
        )
    )"""

# %%
if 1:
    NUM_REGIONS = 1
    model = ci2C.COINForMaskedLM(
        config=ci2C.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            hidden_retention_act="relu",
            #hidden_out_act=None,
            forward_method="parallel",
            apply_decay=True,
            num_decay_parts=1,
            chunkwise_num_chunks=1,
            apply_chunking_globally=False,
            decoder_output="none",
            
            num_regions=NUM_REGIONS,
            decoder_schema=[0, 0, 0, 0, 0, 0],
            cross_encoder_schema=[0, 0, 0, 0, 0, 0],
            multi_head_qkv=False,
            num_heads=1,
            share_S=False,
        )
    )

# %%
if 0:
    NUM_REGIONS = 1
    model = ci2B.COINForMaskedLM(
        config=ci2B.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            #num_labels=NUM_LABELS,
            hidden_retention_act=None,#"relu",
            #hidden_out_act=None,
            forward_method="parallel",
            apply_decay=False,
            num_decay_parts=2,
            chunkwise_num_chunks=1,
            apply_chunking_globally=True,
            apply_hidden_pos_offset=False,
            decoder_output="strict",
            
            num_regions=NUM_REGIONS,
            decoder_schema=[0, 1],
            revert_decoder=True,
            num_repetitions=1,
            block_ioh_schema=None,#[1024, 512, 512, 1024],
            share_S=False,
            apply_softmax_gate=False,
            disable_teacher_forcing=False,

            #group_norm_channels=64,
            num_retention_heads=1,
            xdec_main_switch=False,
            allow_encoder_teacher_forcing=False,
        ),
    )

# %%
if 0:
    model = ci1.COINForMaskedLM(
        config=ci1.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            #num_labels=2 if TEST_SST2 else 7,
        ),
        region_configs=[
            ci1.COINRegionConfig(
                vocab_size=VOCAB_SIZE,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                apply_decay=True,
                num_decay_parts=1,
                forward_method="parallel",
            )
        ] * 2
    )

# %%
if 0:
    model = bounce.BounceForMaskedLM(
        config=bounce.BounceConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            
            apply_decay=True,
            num_decay_parts=1,
            forward_method="parallel",
            hidden_out_act="relu",

            num_regions=2,
            num_bounces=1,
            set_xpos_offset=False,
            apply_decoder_heads=False,
        )
    )

# %%
if 0:
    model = rrb.RRBForBinarySparseLM(
        config=rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,

            apply_decay=True,
            num_decay_parts=1,
            #apply_attention_mask=True
            forward_method="parallel"
        )
    )

# %%
print("{:,}\n{:,}".format(num_parameters(model), num_trainable_parameters(model)))

# %%
try:
    os.mkdir(BASE_PATH)
except OSError as err:
    print(err)
try:
    os.mkdir(CHECKPOINT_PATH)
except OSError as err:
    print(err)
try:
    os.mkdir(WORDPIECE_TOKENIZER_DIR)
except OSError as err:
    print(err)
try:
    os.mkdir(BPE_TOKENIZER_DIR)
except OSError as err:
    print(err)
try:
    os.mkdir(SENTENCE_PIECE_TOKENIZER_DIR)
except OSError as err:
    print(err)
try:
    os.mkdir(DATASET_JSON_PATH)
except OSError as err:
    print(err)

# %%
#dataset = DatasetDict({
#    "train": load_dataset("wikitext", name="wikitext-103-raw-v1", split="train[0:10000]"),
#    "test":  load_dataset("wikitext", name="wikitext-103-raw-v1", split="validation[:1500]")
#})

#dataset

# %%
# oasst1, aaabdon, mcca
DATASET = "oasst1"

HF_TRAIN_ROWS = 25_000
#HF_TRAIN_ROWS = -1
HF_TRAIN_FROM = 0#10_000

HF_TEST_ROWS = 1500
#HF_TEST_ROWS = -1
HF_TEST_FROM = 0

CUSTOM_BASE_DS_PATH = "../datasets/big_AAABDON_Nmax_st200_s0_a10_tvsplit.1_no_norm/"
CUSTOM_DS_TO_FILE = 2

# %%
if DATASET == "mcca":
    train_dataset = load_moses_set({
        "text": [
            "../datasets/multi_cc_aligned_en-de/en/x00[0-2][0-9]",
        ]
    })
    test_dataset = load_moses_set({
        "text": [
            "../datasets/multi_cc_aligned_en-de/en/x800[0-1]",
        ]
    })
    tok_dataset = concatenate_datasets([train_dataset, test_dataset])

elif DATASET == "aaabdon":
    train_ds = [f"{CUSTOM_BASE_DS_PATH}/train/train_00[0-{CUSTOM_DS_TO_FILE}].csv"]
    test_ds = [f"{CUSTOM_BASE_DS_PATH}/validation/validation_00[0-{CUSTOM_DS_TO_FILE}].csv"]
    train_dataset = load_set(train_ds)#.select(list(range(HF_TRAIN_ROWS)))
    test_dataset = load_set(test_ds)#.select(list(range(HF_TEST_ROWS)))
    tok_dataset = load_set([f"{CUSTOM_BASE_DS_PATH}/train/train_*.csv"])
elif DATASET == "oasst1":
    #train_dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", split=f"train[0:{HF_TRAIN_ROWS}]")
    #test_dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", split=f"validation[:{HF_TEST_ROWS}]")
    #tok_dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", split="train")
    
    #train_dataset = load_dataset("QingyiSi/Alpaca-CoT", split=f"train[0:{HF_TRAIN_ROWS}]")  .rename_column("instruction", "text").rename_column("output", "target")
    #test_dataset = load_dataset("QingyiSi/Alpaca-CoT", split=f"test[0:{HF_TEST_ROWS}]")     .rename_column("instruction", "text").rename_column("output", "target")
    #tok_dataset = load_dataset("QingyiSi/Alpaca-CoT")                                       .rename_column("instruction", "text").rename_column("output", "target")

    train_dataset = load_dataset("OpenAssistant/oasst1", split="train").filter(lambda e: e["lang"] == "en")
    test_dataset = load_dataset("OpenAssistant/oasst1", split="validation").filter(lambda e: e["lang"] == "en")
    tok_dataset = load_dataset("OpenAssistant/oasst1", split="train").filter(lambda e: e["lang"] == "en")
    #train_dataset = load_from_disk("../datasets/oasst1/train").filter(lambda e: e["lang"] == "en").select(list(range(HF_TRAIN_ROWS)))
    #test_dataset = load_from_disk("../datasets/oasst1/validation").filter(lambda e: e["lang"] == "en").select(list(range(HF_TEST_ROWS)))
    #tok_dataset = load_from_disk("../datasets/oasst1/train").filter(lambda e: e["lang"] == "en")

    #train_dataset = load_dataset("OpenAssistant/oasst2", split="train").filter(lambda e: e["lang"] == "en").select(list(range(HF_TRAIN_ROWS)))
    #test_dataset = load_dataset("OpenAssistant/oasst2", split="validation").filter(lambda e: e["lang"] == "en").select(list(range(HF_TEST_ROWS)))

if SHUFFLE_TRAIN_DATA:
    print("shuffle")
    train_dataset = train_dataset.shuffle()
if HF_TRAIN_ROWS > 0:
    train_dataset =  train_dataset.select(list(range(HF_TRAIN_FROM, HF_TRAIN_ROWS)))
if SHUFFLE_TEST_DATA:
    print("shuffle")
    test_dataset = test_dataset.shuffle()
if HF_TEST_ROWS > 0:
    test_dataset = test_dataset.select(list(range(HF_TEST_FROM, HF_TEST_ROWS)))

print(train_dataset)
print(test_dataset)
print(tok_dataset)

# %%
train_dataset.to_json(f"{DATASET_JSON_PATH}/train.json")
test_dataset.to_json(f"{DATASET_JSON_PATH}/test.json")

# %%
labels = [label for label in train_dataset.features.keys() if label not in ["text"]]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(label2id)
print(id2label)
print(labels)

# %%
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer, SentencePieceUnigramTokenizer

# %%
if 0:
    #tok_dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", split="train")
    #tok_dataset = load_dataset("glue", name="sst2", split="train")
    #tok_dataset = tok_dataset.rename_column("sentence", "text")
    
    tokenizer = SentencePieceUnigramTokenizer()

    tokenizer.train_from_iterator(
        iterator=tok_dataset["text"], 
        vocab_size=VOCAB_SIZE,
        #min_frequency=2,
        show_progress=True,
        #limit_alphabet=500,
        special_tokens=[
            "<PAD>", 
            "<UNK>", 
            "<CLS>", 
            "<SEP>", 
            "<MASK>"
        ])

    tokenizer = PreTrainedTokenizer(
        tokenizer_object=tokenizer
    )
    tokenizer.save_model(SENTENCE_PIECE_TOKENIZER_DIR)
    #assert False
    #tokenizer = AutoTokenizer.from_pretrained(SENTENCE_PIECE_TOKENIZER_DIR)

# %%
if 1:
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True,
                                        strip_accents=True, lowercase=True)

    tokenizer.train_from_iterator(iterator=tok_dataset["text"], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[
        "[PAD]", 
        "[UNK]", 
        "[CLS]", 
        "[SEP]", 
        "[DOC]",
    #    "[UDOC]",
        "[MASK]"
    ])
    tokenizer.save_model(WORDPIECE_TOKENIZER_DIR)
    #assert False
    tokenizer = BertTokenizer.from_pretrained(WORDPIECE_TOKENIZER_DIR)

# %%
if 0:
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    
    tokenizer.train_from_iterator(iterator=tok_dataset["text"], vocab_size=VOCAB_SIZE, min_frequency=2, length=MAX_POSITION_EMBEDDINGS, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save_model(BPE_TOKENIZER_DIR)
    #assert False
    #tokenizer = RobertaTokenizer.from_pretrained(BPE_TOKENIZER_DIR)

# %%
def encode_and_batch(dataset, tokenizer, max_position_embeddings, batch_size, shuffle=False):
    if ENCODE_CAUSAL_LM:
        encoded = preprocess_for_causallm(dataset, tokenizer, block_size=MAX_POSITION_EMBEDDINGS, remove_columns=dataset.column_names, shift_right=False)
    else:
        encoded = preprocess_for_maskedlm(dataset, tokenizer, max_position_embeddings, remove_columns=dataset.column_names, to_mask=.15, chance_rand_token=.2, 
                                          group_texts=GROUP_TEXTS, mask_token=MASK_TOKEN, pad_token=PAD_TOKEN, sparsify=SPARSIFY, prefix=PREFIX, switch_ii_decoder_ii=SWITCH_II_DECODER_II)
        #encoded = preprocess_for_cot(dataset, tokenizer, max_position_embeddings, remove_columns=dataset.column_names, group_texts=GROUP_TEXTS, pad_token=PAD_TOKEN, sparsify=SPARSIFY, prefix=PREFIX)
       
       
        #encoded = preprocess_for_monologe(dataset, tokenizer, max_position_embeddings, remove_columns=dataset.column_names)
        #encoded = preprocess_for_sparselm(dataset, tokenizer, max_position_embeddings, remove_columns=dataset.column_names)
        #encoded = preprocess_for_binary_sparselm(dataset, tokenizer, max_position_embeddings, remove_columns=dataset.column_names)
        #encoded = preprocess_for_maskedlm(encoded, tokenizer, max_position_embeddings, to_mask=.15, chance_rand_token=.2, group_texts=GROUP_TEXTS, mask_token=MASK_TOKEN)
    print(encoded)
    batched = BatchBuffer(encoded, batch_size)
    if shuffle:
        batched.shuffle()
    print("  finished")
    return batched

# %%
train_loader_call = lambda: encode_and_batch(train_dataset, tokenizer, MAX_POSITION_EMBEDDINGS, BATCH_SIZE, True)
test_loader_call = lambda: encode_and_batch(test_dataset, tokenizer, MAX_POSITION_EMBEDDINGS, BATCH_SIZE)

if not REBATCH:
    train_dataloader = train_loader_call()
    test_dataloader = test_loader_call()

# %%
#encoded_train_dataset = preprocess_for_maskedlm(dataset, tokenizer, MAX_POSITION_EMBEDDINGS, remove_columns=train_dataset.column_names)

# %%
#train_dataloader = BatchBuffer(encoded_dataset["train"], BATCH_SIZE).shuffle()
#test_dataloader = BatchBuffer(encoded_dataset["test"], BATCH_SIZE)

# %%
#batch_schema = list(encoded_dataset["train"].features.keys())
#batch_schema

# %%
def count_item(inp, item):
    count = 0
    total = 0
    for n in inp:
        for r in n:
            i = r
            if not i < 4:
                total += 1
            if i == item:
                count += 1
            #if i != 0 and i != item:
            #    print(i)
    return f"{count} / {total} ; {count/total}"

# %%
print("masked tokens [input_ids]:", count_item(train_dataloader.ds["input_ids"], tokenizer.mask_token_id))
print("masked tokens [labels]:", count_item(test_dataloader.ds["labels"], tokenizer.mask_token_id))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPS)

total_steps = len(train_dataset) / BATCH_SIZE * EPOCHS
warmup_steps = math.ceil(total_steps * 0.05)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

# %%
loss_function = nn.CrossEntropyLoss()

# %%
train_dataloader.schema

# %%
print(train_dataloader.schema)
its = 0
for i, n in enumerate(train_dataloader):
    if i >= its:
        break
    for k in n:
        print(i, "############")
        for l in k:
            print(len(l))
        print("##############")

# %%
model

# %%
%%time
stats = train_bern_model(
    model,
    optimizer,
    scheduler,
    EPOCHS,
    #train_dataloader,
    #test_dataloader,
    #batch_schema,
    device,
    loss_function,
    id2label,
    train_dataloader=train_dataloader if not REBATCH else None,
    test_dataloader=test_dataloader if not REBATCH else None,
    create_train_dataloader=train_loader_call if REBATCH else None,
    create_test_dataloader=test_loader_call if REBATCH else None,
    vocab_size=VOCAB_SIZE,
    print_status=True,
    is_hf_model=IS_HF_MODEL,
    checkpoint_path=CHECKPOINT_PATH,
    batch_size=BATCH_SIZE,
    only_save_core=False,
    epoch_i=EPOCH_I,
    is_encoder_decoder_model=IS_ENCODER_DECODER_MODEL,
    causal_lm=CAUSAL_LM,

    masked_lm_task=True,
    electra_task=False,
    mlm_decode_n=0,
    #mlm_decode_n=.0075,
    #mlm_decode_n=.1,
    mlm_decode_max_chars=200,
    tokenizer=tokenizer,

    dump_coin_regions=False,
    generic_output_class=True,
    #coin_region_lambda=lambda model: model.coin.core.regions
)


# %%
print_stat_tuples(stats)



