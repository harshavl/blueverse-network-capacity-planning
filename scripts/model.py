from pytorch_forecasting import TimeSeriesDataSet

def create_dataset(df, max_encoder_length=30, max_prediction_length=7):
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="bandwidth_mbps_scaled",
        group_ids=["router_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["router_id", "location", "application_type"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["bandwidth_mbps_scaled"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    return training