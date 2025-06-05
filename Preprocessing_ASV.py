import pandas as pd


asv_df = pd.read_csv('ASV_table_clean.csv')

#data dimension 73000 rows and  1168 columns (before transp)
asv_transposed = asv_df.set_index('ASV ID').T

#Many asked by not Import the transposed data already

asv_transposed = (
    asv_transposed
    .reset_index()
    .rename(columns={'index': 'SampleID'})
)

##data dimesion 1168 rows and 2 cols
rfi_df = pd.read_excel('overall_RFI_only.xlsx')



##merging by SampleID
merged = pd.merge(asv_transposed,
                  rfi_df,
                  on='SampleID',
                  how='left')


merged.to_csv('preprocessed_ASV_level_1.csv', index=False)

print("Saved as 'preprocessed_ASV_level_1.csv'")
