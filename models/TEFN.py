import torch
import torch.nn as nn


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        self.means = None
        self.stds = None

    def norm(self, x):
        self.means = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - self.means
        self.stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / self.stds
        return x

    def denorm(self, x):
        x = x * self.stds + self.means
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class EvidenceMachineKernel(nn.Module):
    def __init__(self, C, F, activation=None, use_residual=True):
        super(EvidenceMachineKernel, self).__init__()
        self.C = C
        self.F = 2 ** F
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))
        self.use_residual = use_residual

        # 支持更多非线性激活函数
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = Swish()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'linear':
            self.activation = nn.Linear(self.F, self.F)

    def forward(self, x):
        x = torch.einsum('btc,cf->btcf', x, self.C_weight) + self.C_bias
        if self.activation is not None:
            if self.use_residual:
                x = self.activation(x) + x  # 残差连接
            else:
                x = self.activation(x)
        return x


class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, t_out, c_out):
        # 拼接 t_out 和 c_out
        combined = torch.cat([t_out, c_out], dim=-1)
        q = self.query(combined)
        k = self.key(combined)
        v = self.value(combined)

        scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, v)
        return output


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # 新增消融配置
        self.use_norm = configs.use_norm  # 是否使用归一化层
        self.use_T_model = configs.use_T_model  # 是否使用时间维度模块
        self.use_C_model = configs.use_C_model  # 是否使用通道维度模块
        self.fusion_method = configs.fusion_method  # 融合方式：add/concat/attn
        self.use_probabilistic_layer = configs.use_probabilistic_layer  # 是否使用概率层

        if self.task_name.startswith('long_term_forecast') or self.task_name == 'short_term_forecast':
            if self.use_norm:
                self.nl = NormLayer()
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)

            # 支持非线性的Kernel初始化
            self.T_model = EvidenceMachineKernel(
                self.pred_len + self.seq_len,
                configs.e_layers,
                activation=configs.kernel_activation,
                use_residual=configs.use_residual,  # 新增激活函数配置
            )
            self.C_model = EvidenceMachineKernel(
                configs.enc_in,
                configs.e_layers,
                activation=configs.kernel_activation,
                use_residual=configs.use_residual
            )

            # 新增融合层（用于拼接场景）
            if self.fusion_method == 'concat':
                self.fusion_linear = nn.Linear(2 * (self.pred_len + self.seq_len),
                                               self.pred_len + self.seq_len)
            elif self.fusion_method == 'attn':
                self.attention_fusion = AttentionFusion((self.pred_len + self.seq_len) * 2)

            # 概率层
            if self.use_probabilistic_layer:
                self.probabilistic_layer = nn.Dropout(p=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化层消融点
        if self.use_norm:
            x = self.nl.norm(x_enc)
        else:
            x = x_enc.clone()  # 直接使用原始数据

        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T', C]

        # 时间维度模块消融点
        t_out = self.T_model(x.permute(0, 2, 1)).permute(0, 2, 1, 3) if self.use_T_model else 0

        # 通道维度模块消融点
        c_out = self.C_model(x) if self.use_C_model else 0

        # 灵活融合机制
        if self.fusion_method == 'add':
            fused = t_out + c_out
        elif self.fusion_method == 'concat':
            fused = torch.cat([t_out, c_out], dim=-1)  # [B, T', C, 2F]
            fused = self.fusion_linear(fused.reshape(-1, fused.shape[-2], fused.shape[-1]))  # 拼接后线性变换
            fused = fused.unsqueeze(1)  # 恢复维度
        elif self.fusion_method == 'attn':
            fused = self.attention_fusion(t_out, c_out)

        x = torch.einsum('btcf->btc', fused)

        # 概率层
        if self.use_probabilistic_layer:
            x = self.probabilistic_layer(x)

        # 反归一化
        if self.use_norm:
            x = self.nl.denorm(x)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name.startswith('long_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
