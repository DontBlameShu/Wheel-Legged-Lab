import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class HistoryEncoderActor(MLPModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(128, 64, 32),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    ):
        if obs_set != "actor":
            raise ValueError("HistoryEncoderActor is only intended for the actor.")

        actor_groups = obs_groups[obs_set]
        if actor_groups != ["policy", "policy_history"]:
            raise ValueError(
                f"Expected actor observation groups ['policy', 'policy_history'], got {actor_groups}"
            )

        self.current_obs_dim = obs["policy"].shape[-1]
        self.history_obs_dim = obs["policy_history"].shape[-1]
        self.encoder_latent_dim = 3

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.encoder = MLP(
            self.history_obs_dim,
            self.encoder_latent_dim,
            [128, 64],
            activation,
        )

        mlp_output_dim = (
            self.distribution.input_dim if self.distribution is not None else output_dim
        )

        self.mlp = MLP(
            self.current_obs_dim + self.encoder_latent_dim,
            mlp_output_dim,
            hidden_dims,
            activation,
        )

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(self, obs, masks=None, hidden_state=None):
        current = obs["policy"]
        history = obs["policy_history"]

        combined = torch.cat([current, history], dim=-1)
        combined = self.obs_normalizer(combined)

        current = combined[..., : self.current_obs_dim]
        history = combined[..., self.current_obs_dim :]

        history_latent = self.encoder(history)

        return torch.cat([current, history_latent], dim=-1)