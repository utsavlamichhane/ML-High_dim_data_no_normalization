import pandas as pd


asv_df = pd.read_csv('ASV_table_clean.csv')

#data dimension 73000 rows and  1168 columns (before transp)
asv_transposed = asv_df.set_index('ASV ID').T

# Many asked by not Import the already transposed data. 
# Again, my anser is same we can have 73000 rows but not column.
#  ##so we transpose here in the script so that we can use them as the column.

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
