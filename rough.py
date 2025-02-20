import pandas as pd

# Load the Excel file
df = pd.read_excel("/Users/pavanbandaru/Downloads/cricket-squad-selection/all season cleaned data/allrounderset_ipl.xlsx")

# Function to remove '*' but keep numbers
def clean_column(col):
    return col.astype(str).str.replace(r'(\d+)\*', r'\1', regex=True)

# Apply to specific columns (change 'Column_Name' to your actual column names)
df["HS"] = clean_column(df["HS"])

# Save the cleaned file
df.to_excel("cleaned_file.xlsx", index=False)
