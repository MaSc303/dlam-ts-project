import pandas as pd
import torch

def rollout(
    model,
    frame,
    dynamic_columns,
    static_columns,
    series_mapping,
    forecast_horizon,
    history_length,
    model_horizon,
    device,
    series_scales=None,
):
    if forecast_horizon % model_horizon != 0:
        raise ValueError(
            f"forecast_horizon ({forecast_horizon}) must be divisible by "
            f"model_horizon ({model_horizon})",
        )

    groups=[
        (
            series_id,
            group.sort_values("timestamp").tail(
                history_length+forecast_horizon,
            ),
        )
        for series_id,group in frame.groupby(
            "series_id",
            sort=False,
        )
    ]

    histories=[
        torch.tensor(
            group.iloc[:history_length]["target"].to_numpy(),
            dtype=torch.float32,
        )
        for _,group in groups
    ]

    covariates=[
        torch.tensor(
            group[dynamic_columns].to_numpy(),
            dtype=torch.float32,
        )
        for _,group in groups
    ]

    static_features=[
        torch.tensor(
            group.iloc[0][static_columns].to_numpy(
                dtype="float32",
            ),
            dtype=torch.float32,
        )
        for _,group in groups
    ]

    history=torch.stack(histories).to(device)
    covariates=torch.stack(covariates).to(device)
    static_features=torch.stack(static_features).to(device)

    series_indices=torch.tensor(
        [
            series_mapping[str(series_id)]
            for series_id,_ in groups
        ],
        dtype=torch.long,
        device=device,
    )

    scale=(
        series_scales[series_indices].unsqueeze(1)
        if series_scales is not None
        else None
    )

    predictions=[]
    model.eval()

    with torch.no_grad():
        for offset in range(
            0,
            forecast_horizon,
            model_horizon,
        ):
            past_target=history[:,-history_length:]

            past_covariates=covariates[
                :,
                offset:offset+history_length,
            ]

            future_covariates=covariates[
                :,
                offset+history_length:
                offset+history_length+model_horizon,
            ]

            model_past_target=(
                past_target/scale
                if scale is not None
                else past_target
            )

            pred=model(
                model_past_target,
                past_covariates,
                future_covariates,
                series_indices,
                static_features,
            )

            if scale is not None:
                pred=pred*scale

            predictions.append(pred)
            history=torch.cat(
                [
                    history,
                    pred,
                ],
                dim=1,
            )

    predictions=torch.cat(
        predictions,
        dim=1,
    )[:,:forecast_horizon].cpu()

    rows=[]

    for i,(_,group) in enumerate(groups):
        result=group.tail(
            forecast_horizon,
        )[[
            "series_id",
            "timestamp",
        ]].copy()

        result["prediction"]=predictions[i].numpy()
        rows.append(result)

    return pd.concat(
        rows,
        ignore_index=True,
    )