import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from chatmedgen.common.registry import registry
from chatmedgen.models.base_model import disabled_train
from chatmedgen.models.minigpt_base import MiniGPTBase
from chatmedgen.models.Qformer import BertConfig, BertLMHeadModel

from chatmedgen.models.peft.prefix_tuning import PrefixEncoder

@registry.register_model("chatmedgen")
class chatmedgen(MiniGPTBase):


    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/chatmedgen.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            peft_config=None,      
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            peft_config=peft_config,     
        )

        img_f_dim = self.visual_encoder.num_features * 4
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )


        # if peft_config.prefix_tuning.is_use == True:
        #     self.prefix_tokens = torch.arange(peft_config.prefix_tuning.pre_seq_len).long()
        #     self.prefix_encoder_forPrefixTuning = PrefixEncoder(peft_config.prefix_tuning) 
        #     self.pre_fix_tuning_dropout = torch.nn.Dropout(0.3)

            # self.prefix_encoder_forPrefixTuning.embedding
            



        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()

    def get_prompt_prefix_tuning(self, batch_size, device, config, dtype=torch.half):

        # prefix_encoder = PrefixEncoder(config, device)
        # prefix_tokens = torch.arange(config.pre_seq_len).long()
        prefix_tokens = self.prefix_tokens
        prefix_encoder = self.prefix_encoder_forPrefixTuning

        prefix_tokens = prefix_tokens.to(device)
        # dropout = torch.nn.Dropout(0.1)

        


        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = prefix_encoder(device, prefix_tokens).type(dtype)
        past_key_values = past_key_values.to(device)
        past_key_values = past_key_values.view(
            batch_size,
            config.pre_seq_len,
            config.num_layers * 2,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.pre_fix_tuning_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

            inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg, peft_config=None):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            peft_config=peft_config,          
        )

        ckpt_path = cfg.get("ckpt", "")  
        if ckpt_path:
            print("Load Chat-MedGen Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            # print(ckpt['model'])

        return model



