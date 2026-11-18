import pandas as pd

input_file = "/mnt/d/AI_Seismic_Data/Instance/INSTANCE.csv"
output_file = "/mnt/d/AI_Seismic_Data/Instance/INSTANCE_filtered.csv"

df = pd.read_csv(input_file, low_memory=False)

df["snr_mean"] = df[["trace_E_snr_db", "trace_N_snr_db", "trace_Z_snr_db"]].mean(axis=1)

filtered_df = df[
    (df["source_magnitude"] >=0) & (df["source_magnitude"] <= 1) &
    (df["snr_mean"] >= 20) & (df["snr_mean"] <= 40)
].copy()

filtered_df.drop(columns=["snr_mean"], inplace=True)

filtered_df.to_csv(output_file, index=False)

print(f"Filtered rows: {len(filtered_df)}")
print(f"Saved to {output_file}")