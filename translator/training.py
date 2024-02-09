import os
import sys
import logging
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    CONFIG_MAPPING,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_int8_training
)

from .arguments import ModelArguments, DataTrainingArguments, LoraArguments
from .data import Processor

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)], )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, LoraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, lora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # prepare config, model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # Fix for fp16
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    if model_args.model_name_or_path:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
        )
        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
    else:
        raise NotImplemented

    if base_model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        
    ####################### Load your peft model #######################
    if lora_args.use_lora:
        assert not training_args.gradient_checkpointing, 'Can not use gradients_checkpointing with LoRA'
        
        target_module_dict = {
            "mT5": ['q', 'wi_1', 'k', 'wi_0', 'v', 'wo', 'o', 'lm_head'],
            "T5": ['v', 'q', 'k', 'wi', 'wo', 'o', 'lm_head'],
        }
        target_att_dict = {
            "T5": ['v', 'q', 'k', 'o'],
        }

        if lora_args.linear_layer:
            if lora_args.att_blocks:
                target_modules = target_att_dict['T5']
            else:
                if ("mt5" or "flan-t5") in model_args.model_name_or_path:
                    target_modules = target_module_dict['mT5']
                elif "t5" in model_args.model_name_or_path:
                    target_modules = target_module_dict['T5']

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            target_modules = target_modules,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type=TaskType.SEQ_CLS
        )
        
        # prepare int-8 model for training
        if lora_args.use_int8_training:
            base_model = prepare_model_for_int8_training(base_model)

        # add LoRA adaptor
        model = get_peft_model(base_model, lora_config)
        print('-' * 50, '\n')
        print_trainable_parameters(base_model)
        print('-' * 50, '\n')
    else:
        model = base_model

    processor = Processor(
        tokenizer, 
        training_args.per_device_train_batch_size, 
        data_args, 
        training_args.seed
    ).__call__()

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            max_length=data_args.max_len
        )

    cls_trainer = Seq2SeqTrainer
    trainer = cls_trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=processor['train'],
        eval_dataset=processor['validation'],
    )
        
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # Training
    trainer.train()

    # Save Model
    trainer.model.save_pretrained(training_args.output_dir)
    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    main()