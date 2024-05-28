# %%
import glob
import time
import os
import datetime
import math
import datasets
from datasets import DatasetDict, load_dataset, Dataset
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch import nn
import transformers
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    T5ForSequenceClassification,
    T5Config,
    BertTokenizer, 
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load_set import load_set
from epoch_stats import EpochStats, print_stat_tuples
import model_training
from model_training import (
    BatchBuffer, 
    train_bern_model, 
    mask_tokens, 
    preprocess_with_given_labels, 
    num_parameters, 
    num_trainable_parameters, 
    preprocess_for_causallm, 
    preprocess_for_multiple_choice,
    preprocess_for_seq2seq_swag
)
import bert_i1_1_modeling as bert_i1_1
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
TO_FILE = 1
BATCH_SIZE = 8
CHECKPOINT_PATH = None # datetime.datetime.now().strftime("tmp_models/rann_sffn/run_part_load_%Y-%m-%d_%H:%M:%S")
USE_CUSTOM_DATALOADER = True
SHUFFLE_CUSTOM_DATALOADER = True
LEARNING_RATE = 1e-5
EPS = 1e-8
EPOCHS = 10

# %%
CHECKPOINT_PATH

# %%
# bigger vocab_size in tokenizer + smaller vocab_size in model -> better

# %%
VOCAB_SIZE = 30_522
#VOCAB_SIZE = 32_000
#VOCAB_SIZE = 52_000
MAX_POSITION_EMBEDDINGS = 512
IS_HF_MODEL = False
GENERIC_OUTPUT_CLASS = True

# %%
# ALWAYS CHECK num_labels, RFFN doesn't throw an error on a wrong parameter
rffn_base_model_path = "tmp_models/COIN-i2C_oasst1_25k_1x6_0dec-none_parallel_no-decay/"
#rffn_base_model_path = "tmp_models/COIN-i2B_oasst1_25k_1x3_000dec_none-decoder-revert-out_chunkwise_nt-case-3_2-decay-parts_allow-enc-tf/"
#rffn_base_model_path = "tmp_models/RRB_oasst1_25k_2-2-encoder_0-1-decoder_decay_maskedLM.15_.2share_docx1_wtf/"
#rffn_tokenizer_path = "pretrained_models/rffn_wikitext_516_tokenizer"
rffn_tokenizer_path = rffn_base_model_path

# %%
# default, sst2, swag, uni-main-hyp
TEST_METHOD = "default"
ILOC_LIMIT = None
DEFAULT_TEACHER_FORCING = False

# %%
if TEST_METHOD == "default":
    NUM_LABELS = 7
elif TEST_METHOD == "sst2":
    NUM_LABELS = 2
elif TEST_METHOD == "swag":
    NUM_LABELS = 4
elif TEST_METHOD == "uni-main-hyp":
    NUM_LABELS = 10

TEST_SST2 = TEST_METHOD == "sst2"
ONE_LABEL_ONLY = TEST_SST2

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = mdrb.MDRBForSequenceClassification(
        config=mdrb.MDRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,

            apply_decay=True,
            num_decay_parts=1
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = smpl_rrb.RRBForSequenceClassification(
        config=smpl_rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,

            apply_decay=True,
            num_decay_parts=1,
            #hidden_size=36*36*2,
            forward_method="parallel",
            copy_first_decoder_ids=True,

            block_schema=[[2], [4], [4], [2]],
            block_decoder_schema=[0, 0, 1, 1],
            #layer_decoder_schema=[0, 1],
            #layer_ffn_schema=[1, 1],
            #apply_hierarchical_ffn=True,
            collector_method=4,
            extend_X=True,
            sum_in_qkv=False,
            qkv_sum_slice=False,
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = smpl_rrb.RRBForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_4/model",
        num_labels=2 if TEST_SST2 else 7,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        vocab_size=VOCAB_SIZE,

        #apply_layer_wise_ffn=False
        #block_schema=(2, 2),
        #block_decoder_schema=(0, 0, 0, 0),
        #layer_decoder_schema=(0, 0, 0, 0)
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = mdrb.MDRBForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_3/model",
        num_labels=2 if TEST_SST2 else 7,
    )


# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = srb.RRBForSequenceClassification(
        config=srb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,

            apply_decay=True,
            num_decay_parts=1,
            #hidden_size=36*36*2,
            forward_method="parallel",
            copy_first_decoder_ids=True,
            #apply_dense_gate=False,
            decoder_pipeline=False,
            
            block_schema=[2, 2],
            block_decoder_schema=[0, 1],
            #layer_decoder_schema=[0, 1],
            #apply_hierarchical_ffn=True,

            #hidden_size=32*32,
            #num_retention_heads=32*32,
            hidden_out_act="relu",
            stack_QKV=True,
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = srb.RRBForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_0/model",
        num_labels=2 if TEST_SST2 else 7,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        vocab_size=VOCAB_SIZE,

        #apply_layer_wise_ffn=False
        #block_schema=(2, 2),
        #block_decoder_schema=(0, 0, 0, 0),
        #layer_decoder_schema=(0, 0, 0, 0)
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rbsrm.RBSRMForSequenceClassification(
        config=rbsrm.RBSRMConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,

            apply_decay=True,
            num_decay_parts=1,
            #hidden_size=36*36*2,
            forward_method="parallel",
            
            block_schema=[6, 6],
            decoder_schema=[0, 1],
            K_schema=[3, 3],
            #apply_block_wise_ffn=True,

            hidden_out_act="relu",
            hidden_retention_act="relu",
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rbsrm.RBSRMForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_5/model",
        num_labels=2 if TEST_SST2 else 7,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        vocab_size=VOCAB_SIZE,
    )

# %%
CHECK_RUN = False

# %%
if 1:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    NUM_REGIONS = 1
    model = ci2C.COINForSequenceClassification(
        config=ci2C.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=NUM_LABELS,
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

            print_checks=CHECK_RUN,
        )
    )

# %%
if 0:
    n_iter = 1
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = ci2C.COINForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_{n_iter}/model/",
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        num_labels=NUM_LABELS,

        #region_config=ci2B.COINRegionConfig.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/model/")       
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    NUM_REGIONS = 1
    model = ci2B.COINForSequenceClassification(
        config=ci2B.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=NUM_LABELS,
            hidden_retention_act=None,#"relu",
            #hidden_out_act=None,
            forward_method="parallel",
            apply_decay=True,
            num_decay_parts=2,
            chunkwise_num_chunks=2,
            apply_chunking_globally=False,
            apply_hidden_pos_offset=False,
            decoder_output="none",
            
            num_regions=NUM_REGIONS,
            decoder_schema=[0, 0],
            revert_decoder=True,
            num_repetitions=1,
            block_ioh_schema=None,#[1024, 512, 512, 1024],
            share_S=False,
            apply_softmax_gate=False,
            disable_teacher_forcing=False,
            R_skip_connection=False,

            #group_norm_channels=64,
            num_retention_heads=1,
            allow_encoder_teacher_forcing=False,

            print_checks=CHECK_RUN,
        ),
    )

# %%
if 0:
    n_iter = 5
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = ci2B.COINForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_{n_iter}/model/",
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        num_labels=NUM_LABELS,

        #region_config=ci2B.COINRegionConfig.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/model/")       
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = ci2B.COINForMajoritySequenceClassification(
        config=ci2B.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=NUM_LABELS,

            num_regions=2,
        ),
        region_config=ci2B.COINRegionConfig(
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
                decoder_schema=[0, 0],
            )
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
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = ci2A.COINForSequenceClassification(
        config=ci2A.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=NUM_LABELS,
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
                decoder_schema=[0, 1],
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
if 0:
    n_iter = 3
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    #model = ci2A.COINForSequenceClassification(
    model = ci2A.COINForSequenceClassification.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/model/",
        #config=ci2A.COINConfig(
        #    vocab_size=VOCAB_SIZE,
        #    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=NUM_LABELS,
        #),
        regions=[
            ci2A.COINRegion.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/regions/region_0/"),
            ci2A.COINRegion.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/regions/region_1/"),
            ci2A.COINRegion.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/regions/region_2/"),
            ci2A.COINRegion.from_pretrained(f"{rffn_base_model_path}/model/epoch_{n_iter}/regions/region_3/"),
        ],
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
        ),
    )"""


# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = ci2A.COINForMultipleChoice(
        config=ci2A.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,
            #num_labels=10,
        ),
        regions=[
            ci2A.COINRegionConfig(
                vocab_size=VOCAB_SIZE,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                apply_decay=True,
                num_decay_parts=1,
                forward_method="parallel",
                num_layers=1,
                is_decoder=True,
                #hidden_retention_act=None,
                #hidden_out_act=None,
                reverse_decay=True,
            )
        ] * 4,
        encoder=None,
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = ci2A.COINForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_9/model",
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        num_labels=2 if TEST_SST2 else 7,
        region_configs=[
            ci2A.COINRegionConfig(
                vocab_size=VOCAB_SIZE,
                max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                apply_decay=True,
                num_decay_parts=1,
                forward_method="parallel",
                #hidden_retention_act=None,
                #hidden_out_act=None,
                reverse_decay=True,
            )
        ] * 2
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = coin.COINForSequenceClassification(
        config=coin.COINConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,
        ),
        region_configs=[
            coin.COINRegionConfig(
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
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = coin.COINForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_6/model",
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        num_labels=2 if TEST_SST2 else 7,
        region_configs=[
            coin.COINRegionConfig(
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
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = bounce.BounceForSequenceClassification(
        config=bounce.BounceConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=2 if TEST_SST2 else 7,

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
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = bounce.BounceForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_6/model",
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        num_labels=2 if TEST_SST2 else 7,
        num_bounces=2,
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rrb.RRBForSequenceClassification(
        config=rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_labels=NUM_LABELS,

            apply_decay=True,
            num_decay_parts=1,
            #hidden_size=36*36*2,
            forward_method="parallel",
            copy_first_decoder_ids=True,
            apply_dense_gate=False,
            decoder_pipeline=False,
            
            block_schema=[2, 2],
            block_decoder_schema=[0, 1],
            #layer_decoder_schema=[0, 1],
            #apply_hierarchical_ffn=True,
            #apply_block_wise_ffn=True,
            #ffn_intermediate_factor=2.,

            #hidden_size=32*32,
            #num_retention_heads=32*32,
            #hidden_retention_act="softplus",
            #hidden_out_act="relu",
            num_repetitions=1,
            apply_decoder_heads=False,
            #apply_dense_gate=True,

            #hidden_retention_act=None,
            #hidden_out_act=None,
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rrb.RRBForSequenceClassification.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_0/model",
        num_labels=2 if TEST_SST2 else 7,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        vocab_size=VOCAB_SIZE,

        #apply_layer_wise_ffn=False
        #block_schema=(2, 2),
        #block_decoder_schema=(0, 0, 0, 0),
        #layer_decoder_schema=(0, 0, 0, 0)
        apply_decoder_heads=False,
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rrb.RRBForMultipleChainthrough(
        config=rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            #num_labels=2 if TEST_SST2 else 7,

            apply_decay=True,
            num_decay_parts=1,
            #hidden_size=36*36*2,
            forward_method="parallel",
            copy_first_decoder_ids=True,
            #apply_dense_gate=False,
            decoder_pipeline=False,
            
            block_schema=[4, 4],
            block_decoder_schema=[1, 1],
            #layer_decoder_schema=[0, 1],
            #apply_hierarchical_ffn=True,
            #apply_block_wise_ffn=True,
            #ffn_intermediate_factor=2.,

            #hidden_size=32*32,
            #num_retention_heads=32*32,
            #hidden_retention_act="softplus",
            hidden_out_act="relu",
            num_repetitions=1,
            apply_decoder_heads=False,
            #apply_dense_gate=True,
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rrb.RRBForMaskedLM(
        config=rrb.RRBConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            #num_labels=2 if TEST_SST2 else 7,

            apply_decay=True,
            num_decay_parts=1,
            #hidden_size=36*36*2,
            forward_method="parallel",
            copy_first_decoder_ids=True,
            #apply_dense_gate=False,
            decoder_pipeline=False,
            
            block_schema=[2, 2],
            block_decoder_schema=[0, 1],
            #layer_decoder_schema=[0, 1],
            #apply_hierarchical_ffn=True,
            #apply_block_wise_ffn=True,
            #ffn_intermediate_factor=2.,

            #hidden_size=32*32,
            #num_retention_heads=32*32,
            #hidden_retention_act="softplus",
            hidden_out_act="relu",
            num_repetitions=1,
            apply_decoder_heads=False,
            #apply_dense_gate=True,
        )
    )

# %%
if 0:
    tokenizer = BertTokenizer.from_pretrained(f"{rffn_base_model_path}/wordpiece_tokenizer/")
    model = rrb.RRBForMultipleChoice.from_pretrained(
        f"{rffn_base_model_path}/model/epoch_0/model",
        #num_labels=2 if TEST_SST2 else 7,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        vocab_size=VOCAB_SIZE,

        #apply_layer_wise_ffn=False
        #block_schema=(2, 2),
        #block_decoder_schema=(0, 0, 0, 0),
        #layer_decoder_schema=(0, 0, 0, 0)
        apply_decoder_heads=False,
    )

# %%
if 0:
    IS_HF_MODEL = True
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = transformers.RobertaForMultipleChoice.from_pretrained("roberta-base")

# %%
if 0:
    IS_HF_MODEL = True
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.BertForMultipleChoice(config=transformers.BertConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    ))

# %%
if 0:
    IS_HF_MODEL = True
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        #"microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_labels=NUM_LABELS,
    )

# %%
if 0:
    IS_HF_MODEL = True
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=NUM_LABELS
    )

# %%
if 0:
    IS_HF_MODEL = True
    tokenizer = transformers.DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    model = transformers.DebertaForSequenceClassification.from_pretrained(
        "microsoft/deberta-base",
        num_labels=NUM_LABELS,
    )

# %%
print("{:,}\n{:,}".format(num_parameters(model), num_trainable_parameters(model)))

# %%
base_ds_path = "../datasets/big_AAABDON_Nmax_st200_s0_a10_tvsplit.1_no_norm/"

train_ds = [
    f"{base_ds_path}/train/train_00[0-{TO_FILE}].csv"

    #f"datasets/big_AAABDON_Nmax_st200_s0_a10_tvsplit.1_no_norm/train/train_00[0-9].csv",
    #f"datasets/big_AAABDON_Nmax_st200_s0_a10_tvsplit.1_no_norm/train/train_01[0-9].csv"
]
test_ds = [
    f"{base_ds_path}/validation/validation_00[0-{TO_FILE}].csv"

    #f"datasets/big_AAABDON_Nmax_st200_s0_a10_tvsplit.1_no_norm/validation/validation_00[0-9].csv",
    #f"datasets/big_AAABDON_Nmax_st200_s0_a10_tvsplit.1_no_norm/validation/validation_01[0-9].csv"
]

# %%
if TEST_METHOD == "sst2":
    dataset = DatasetDict({
        "train": load_dataset("glue", name="sst2", split="train[:10000]").rename_column("sentence", "text"),
        "test": load_dataset("glue", name="sst2", split="validation").rename_column("sentence", "text")
    })
elif TEST_METHOD == "default":
    dataset = DatasetDict({
        "train": load_set(train_ds, unused_fields=["author", "subreddit", "style"], iloc_limit=ILOC_LIMIT),
        "test":  load_set(test_ds, unused_fields=["author", "subreddit", "style"], iloc_limit=ILOC_LIMIT)

        #"train": load_dataset("glue", name="mnli", split="train[0:10000]"),
        #"test": load_dataset("glue", name="mnli", split="validation_matched[0:1500]")

        #"train": load_dataset("squad_v2", split="train[0:10000]"),
        #"test": load_dataset("squad_v2", split="test[0:1500]")
    })
elif TEST_METHOD == "swag":
    dataset = DatasetDict({
        "train": load_dataset("Rowan/hellaswag", split="train"),
        "test": load_dataset("Rowan/hellaswag", split="validation")
    })
elif TEST_METHOD == "uni-main-hyp":
    dataset = DatasetDict({
        "train": load_set(["../uni-hyp-class/wordpiece_abstracts_train.csv"], unused_fields=["head", "body", "strlabels"]),
        "test": load_set(["../uni-hyp-class/wordpiece_abstracts_test.csv"], unused_fields=["head", "body", "strlabels"]),
    })
else:
    raise ValueError(TEST_METHOD)
dataset

# %%
if ONE_LABEL_ONLY:
    #labels = np.unique(train_df["label"]).tolist()
    labels = np.unique(dataset["train"]["label"]).tolist()
else:
    labels = [label for label in dataset['train'].features.keys() if label not in ["text"]]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(label2id)
print(id2label)
print(labels)

# %%
if TEST_METHOD in ("default", "sst2", "uni-main-hyp"):
    encoded_dataset = preprocess_with_given_labels(dataset, tokenizer, labels, label2id, MAX_POSITION_EMBEDDINGS, ONE_LABEL_ONLY, remove_columns=dataset["train"].column_names, 
                                                   default_teacher_forcing=DEFAULT_TEACHER_FORCING)
elif TEST_METHOD == "swag":
    #encoded_dataset = preprocess_for_multiple_choice(dataset, tokenizer, MAX_POSITION_EMBEDDINGS, remove_columns=dataset["train"].column_names, num_proc=4)
    encoded_dataset = preprocess_for_seq2seq_swag(dataset, tokenizer, MAX_POSITION_EMBEDDINGS, remove_columns=dataset["train"].column_names, num_proc=4)
encoded_dataset

# %%
batch_schema = list(encoded_dataset["train"].features.keys())
batch_schema

# %%
if USE_CUSTOM_DATALOADER:
    #train_dataloader = create_dataloader(encoded_dataset["train"])
    #test_dataloader = create_dataloader(encoded_dataset["test"])
    train_dataloader = BatchBuffer(encoded_dataset["train"], BATCH_SIZE)
    if SHUFFLE_CUSTOM_DATALOADER:
        train_dataloader.shuffle()
    test_dataloader = BatchBuffer(encoded_dataset["test"], BATCH_SIZE)
else:
    USE_TOKEN_TYPE_IDS = "token_type_ids" in encoded_dataset["train"].features
    USE_DEC_II = "decoder_input_ids" in encoded_dataset["train"].features
    # Load input data into tensors
    train_input_ids = torch.tensor(encoded_dataset["train"]["input_ids"])
    if USE_TOKEN_TYPE_IDS:
        train_token_type_ids = torch.tensor(encoded_dataset["train"]["token_type_ids"])
    train_masks = torch.tensor(encoded_dataset["train"]["attention_mask"])
    train_labels = torch.tensor(encoded_dataset["train"]["labels"])
    if USE_DEC_II:
        train_dec_ii = torch.tensor(encoded_dataset["train"]["decoder_input_ids"])

    test_input_ids = torch.tensor(encoded_dataset["test"]["input_ids"])
    if USE_TOKEN_TYPE_IDS:
        test_token_type_ids = torch.tensor(encoded_dataset["test"]["token_type_ids"])
    test_masks = torch.tensor(encoded_dataset["test"]["attention_mask"])
    test_labels = torch.tensor(encoded_dataset["test"]["labels"])
    if USE_DEC_II:
        test_dec_ii = torch.tensor(encoded_dataset["test"]["decoder_input_ids"])

    # Create the DataLoader and Sampler for both sets.
    if USE_TOKEN_TYPE_IDS:
        train_data = TensorDataset(train_input_ids, train_token_type_ids, train_masks, train_labels)
    else:
        train_data = TensorDataset(train_input_ids, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, 
        sampler=train_sampler, 
        batch_size=BATCH_SIZE)

    if USE_TOKEN_TYPE_IDS:
        test_data = TensorDataset(test_input_ids, test_token_type_ids, test_masks, test_labels)
    else:
        test_data = TensorDataset(test_input_ids, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, 
        sampler=test_sampler, 
        batch_size=BATCH_SIZE)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPS)

total_steps = len(train_dataloader) * EPOCHS
warmup_steps = math.ceil(total_steps * 0.05)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

if len(labels) <= 2:
    loss_function = nn.CrossEntropyLoss()
else:
    loss_function = nn.BCEWithLogitsLoss()

# %%
loss_function

# %%
if CHECKPOINT_PATH is not None:
    try:
        os.mkdir(CHECKPOINT_PATH)
    except OSError as err:
        print(err)

# %%
#len(encoded_dataset["train"]["input_ids"]), len(encoded_dataset["train"]["input_ids"][0])

# %%
batch_schema

# %%
model

# %%
model.config

# %%
%%time
stats = train_bern_model(
    model,
    optimizer,
    scheduler,
    EPOCHS,
    device,
    loss_function,
    id2label,
    batch_schema=batch_schema,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    vocab_size=VOCAB_SIZE,
    print_status=True,
    is_hf_model=IS_HF_MODEL,
    checkpoint_path=CHECKPOINT_PATH,
    batch_size=BATCH_SIZE,
    only_save_core=False,
    one_label_only=ONE_LABEL_ONLY,
    mixed_lm_task=False,
    mixed_lm_loss_function=nn.CrossEntropyLoss(),
    
    generic_output_class=True,
    
    #forward_args=["input_ids", "token_type_ids", "attention_mask", "labels"],
    #forward_args=["input_ids"],

    add_layers_on_stagnation=False,
    num_layers_to_add=1,
    add_layers_threshold=0.01, #0.005,
    plot_k_topics=False,

    batch_hack_train=True,
    mlm_decode_n=0,#.0075,
    tokenizer=tokenizer,

    masked_lm_task=False,
    check_run=CHECK_RUN,
    retain_graph=False,
)

# %%
## SST 2 test
## [loss] / [acc/ham]

# 35k
# .61 / .79 epoch 4 ; .59 / .78 epoch 3 ; .52 / .776 epoch 2 15k pretraining, num_labels=2

# .48 / .80 epoch 3


# 10k
# .51 / .758 epoch 2 empty
# .57 / .778 epoch 3 empty

# %%
# hf bert (empty): batch_size=6, time per epoch=6:30min, 5734MiB VRAM, 0.756 0.786 epoch 6
# hf t5 (empty): batch_size=6, time per epoch=5:17min, 5306MiB VRAM, 0.758 0.773 epoch 5
# hf t5 (empty) num_layers=12 num_heads=12: batch_size=3, time per epoch=11:45min, 7416MiB VRAM, 0.758 0.787 epoch 5



