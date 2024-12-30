from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn

from peft import inject_adapter_in_model, LoraConfig
from diffusers.training_utils import cast_training_params

from copy import deepcopy

class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = deepcopy(pipe.unet)
        self.unet_lora = deepcopy(pipe.unet)

        #--------------------------------
        # LoraConfig
        self.lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        #--------------------------------

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def use_lora(self):
        # if self.unet == self.unet_lora:
        #     print("aue")
        #     raise Exception
        self.unet.requires_grad_(False)
        for param in self.unet.parameters():
            param.requires_grad_(False)


        for param in self.unet_lora.parameters():
            param.requires_grad_(False)

        # for param in self.lora_config.parameters():
        #     param.requires_grad_(True)
        # print(self.lora_config)

        self.unet_lora = inject_adapter_in_model(self.lora_config, self.unet_lora)
        # lora_layers = filter(lambda p: p.requires_grad, self.unet_lora.parameters())
        # if self.args.precision == "fp16":
        # # only upcast trainable parameters (LoRA) into fp32
        #     cast_training_params(self.unet_lora, dtype=torch.float32)

        return self.unet_lora.parameters()

    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred

    def get_noise_preds_lora(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet_lora(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for SDS

        t = torch.randint(1, 999, (1, ), device=self.device)

        eps = torch.randn(latents.shape, device=self.device)

        x_t = latents * torch.sqrt(self.alphas[t]) + eps * torch.sqrt(1 - self.alphas[t])
        
        pred_eps = self.get_noise_preds(x_t, t + 1, text_embeddings, guidance_scale)

        loss = torch.mean((pred_eps - eps) ** 2)

        grad = torch.autograd.grad(loss, pred_eps, retain_graph=True)[0].detach()

        return torch.mean(grad * x_t)

    
    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for PDS
        t = torch.randint(self.min_step, self.max_step, (1, ), device=self.device)

        eps_t = torch.randn(src_latents.shape, device=self.device)
        eps_t_minusone = torch.randn(src_latents.shape, device=self.device)

        x_t_src = src_latents * torch.sqrt(self.alphas[t]) + eps_t * torch.sqrt(1 - self.alphas[t])
        x_t_tgt = tgt_latents * torch.sqrt(self.alphas[t]) + eps_t * torch.sqrt(1 - self.alphas[t])

        x_t_minusone_src = src_latents * torch.sqrt(self.alphas[t - 1]) + eps_t_minusone * torch.sqrt(1 - self.alphas[t - 1])
        x_t_minusone_tgt = tgt_latents * torch.sqrt(self.alphas[t - 1]) + eps_t_minusone * torch.sqrt(1 - self.alphas[t - 1])

        sigma_t = torch.sqrt((1 - self.alphas[t - 1])/(1 - self.alphas[t])) * torch.sqrt(1 - self.alphas[t] / self.alphas[t - 1])

        
        pred_eps_src = self.get_noise_preds(x_t_src, t, src_text_embedding, guidance_scale)
        pred_eps_tgt = self.get_noise_preds(x_t_tgt, t, tgt_text_embedding, guidance_scale)

        mean_src = torch.sqrt(self.alphas[t - 1]) * src_latents + torch.sqrt(1 - self.alphas[t - 1] - sigma_t ** 2) * pred_eps_src
        mean_tgt = torch.sqrt(self.alphas[t - 1]) * tgt_latents + torch.sqrt(1 - self.alphas[t - 1] - sigma_t ** 2) * pred_eps_tgt

        z_src = (x_t_minusone_src - mean_src) / sigma_t
        z_tgt = (x_t_minusone_tgt - mean_tgt) / sigma_t

        loss = torch.mean(((z_src - z_tgt) ** 2))

        grad = torch.autograd.grad(loss, pred_eps_tgt, retain_graph=True)[0].detach()

        return torch.mean(grad * x_t_tgt)

    
    def get_vds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        t = torch.randint(1, 999, (1, ), device=self.device)

        eps = torch.randn(latents.shape, device=self.device)

        x_t = latents * torch.sqrt(self.alphas[t]) + eps * torch.sqrt(1 - self.alphas[t])
        
        pred_eps_wo_lora = self.get_noise_preds(x_t, t + 1, text_embeddings, guidance_scale)
        pred_eps_w_lora = self.get_noise_preds_lora(x_t, t + 1, text_embeddings, guidance_scale)

        loss_vds = torch.mean((pred_eps_wo_lora - pred_eps_w_lora) ** 2)

        grad_vds = torch.autograd.grad(loss_vds, pred_eps_w_lora, retain_graph=True)[0].detach()

        res_grad_vds = torch.mean(grad_vds * x_t)

        # res_loss_lora = torch.mean((pred_eps_w_lora - eps) ** 2)

        return res_grad_vds, eps, t


    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents



class StableDiffusionLora(nn.Module):
    def __init__(self, args, use_lora=False, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.unet_lora = pipe.unet

        #--------------------------------
        # LoraConfig
        # self.lora_config = LoraConfig(
        #     r=4,
        #     lora_alpha=4,
        #     init_lora_weights="gaussian",
        #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        # )

        if use_lora:
            self.lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            

        #--------------------------------

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def use_lora(self):
        self.unet.requires_grad_(False)

        for param in self.unet_lora.parameters():
            param.requires_grad_(False)
        self.unet_lora = inject_adapter_in_model(self.lora_config, deepcopy(self.unet_lora))

        # print(self.lora_config)

        # self.unet_lora = inject_adapter_in_model(self.lora_config, self.unet_lora)
        # lora_layers = filter(lambda p: p.requires_grad, self.unet_lora.parameters())

        return self.unet_lora.parameters()

    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred

    def get_noise_preds_lora(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet_lora(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for SDS

        t = torch.randint(1, 999, (1, ), device=self.device)

        eps = torch.randn(latents.shape, device=self.device)

        x_t = latents * torch.sqrt(self.alphas[t]) + eps * torch.sqrt(1 - self.alphas[t])
        
        pred_eps = self.get_noise_preds(x_t, t + 1, text_embeddings, guidance_scale)

        loss = torch.mean((pred_eps - eps) ** 2)

        grad = torch.autograd.grad(loss, pred_eps, retain_graph=True)[0].detach()

        return torch.mean(grad * x_t)

    
    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for PDS
        t = torch.randint(self.min_step, self.max_step, (1, ), device=self.device)

        eps_t = torch.randn(src_latents.shape, device=self.device)
        eps_t_minusone = torch.randn(src_latents.shape, device=self.device)

        x_t_src = src_latents * torch.sqrt(self.alphas[t]) + eps_t * torch.sqrt(1 - self.alphas[t])
        x_t_tgt = tgt_latents * torch.sqrt(self.alphas[t]) + eps_t * torch.sqrt(1 - self.alphas[t])

        x_t_minusone_src = src_latents * torch.sqrt(self.alphas[t - 1]) + eps_t_minusone * torch.sqrt(1 - self.alphas[t - 1])
        x_t_minusone_tgt = tgt_latents * torch.sqrt(self.alphas[t - 1]) + eps_t_minusone * torch.sqrt(1 - self.alphas[t - 1])

        sigma_t = torch.sqrt((1 - self.alphas[t - 1])/(1 - self.alphas[t])) * torch.sqrt(1 - self.alphas[t] / self.alphas[t - 1])

        
        pred_eps_src = self.get_noise_preds(x_t_src, t, src_text_embedding, guidance_scale)
        pred_eps_tgt = self.get_noise_preds(x_t_tgt, t, tgt_text_embedding, guidance_scale)

        mean_src = torch.sqrt(self.alphas[t - 1]) * src_latents + torch.sqrt(1 - self.alphas[t - 1] - sigma_t ** 2) * pred_eps_src
        mean_tgt = torch.sqrt(self.alphas[t - 1]) * tgt_latents + torch.sqrt(1 - self.alphas[t - 1] - sigma_t ** 2) * pred_eps_tgt

        z_src = (x_t_minusone_src - mean_src) / sigma_t
        z_tgt = (x_t_minusone_tgt - mean_tgt) / sigma_t

        loss = torch.mean(((z_src - z_tgt) ** 2))

        grad = torch.autograd.grad(loss, pred_eps_tgt, retain_graph=True)[0].detach()

        return torch.mean(grad * x_t_tgt)

    

    def get_vds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        t = torch.randint(1, 999, (1, ), device=self.device)

        eps = torch.randn(latents.shape, device=self.device)

        x_t = latents * torch.sqrt(self.alphas[t]) + eps * torch.sqrt(1 - self.alphas[t])

        pred_eps_wo_lora = self.get_noise_preds(x_t, t + 1, text_embeddings, guidance_scale)
        pred_eps_w_lora = self.get_noise_preds_lora(x_t, t + 1, text_embeddings, guidance_scale)


        loss = torch.sum((pred_eps_wo_lora - pred_eps_w_lora) ** 2)
        # if loss == 0.:
        #     print('here')
        grad_vds = (pred_eps_wo_lora - pred_eps_w_lora).detach()

        return torch.sum(grad_vds * latents), eps, t

    def get_lora_loss(
        self,
        latents, 
        eps,
        t,
        text_embeddings,
        guidance_scale=100,
        grad_scale=1
    ):

        x_t = latents * torch.sqrt(self.alphas[t]) + eps * torch.sqrt(1 - self.alphas[t])

        # pred_eps_wo_lora = self.get_noise_preds(x_t, t + 1, text_embeddings, guidance_scale)
        pred_eps_w_lora = self.get_noise_preds_lora(x_t, t + 1, text_embeddings, guidance_scale)

        loss = torch.mean((pred_eps_w_lora - eps) ** 2)

        return loss


    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

