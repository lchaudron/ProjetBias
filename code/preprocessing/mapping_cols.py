import pandas as pd
import numpy as np

df = pd.read_csv('data/brute/data.csv', sep=';')

df['Target'] = np.where(df['Target'] == 'Dropout', 1, 0)


father_occupation_mapping = {
    # 0 - Students & Others
    0: "Student/Other", 90: "Student/Other", 99: "Student/Other",
    
    # 1 - Management
    1: "Management", 112: "Management", 114: "Management",
    
    # 2 - High-Level Specialists (STEM, Health, Education)
    2: "High-Level Specialists", 121: "High-Level Specialists", 
    122: "High-Level Specialists", 123: "High-Level Specialists", 124: "High-Level Specialists",
    
    # 3 - Tech & Associate Professionals
    3: "Intermediate/Tech", 131: "Intermediate/Tech", 132: "Intermediate/Tech", 
    134: "Intermediate/Tech", 135: "Intermediate/Tech",
    
    # 4 - Administrative Support
    4: "Administrative", 141: "Administrative", 143: "Administrative", 144: "Administrative",
    
    # 5 - Service & Sales
    5: "Service/Sales", 151: "Service/Sales", 152: "Service/Sales", 
    153: "Service/Sales", 154: "Service/Sales",
    
    # 6 - Skilled Agri & Fishery
    6: "Skilled Agri/Fishery", 161: "Skilled Agri/Fishery", 163: "Skilled Agri/Fishery",
    
    # 7 - Craft & Skilled Trades
    7: "Skilled Trades", 171: "Skilled Trades", 172: "Skilled Trades", 
    174: "Skilled Trades", 175: "Skilled Trades",
    
    # 8 - Operators & Assembly
    8: "Operators/Assembly", 181: "Operators/Assembly", 182: "Operators/Assembly", 183: "Operators/Assembly",
    
    # 9 - Elementary/Unskilled
    9: "Unskilled/Elementary", 192: "Unskilled/Elementary", 193: "Unskilled/Elementary", 
    194: "Unskilled/Elementary", 195: "Unskilled/Elementary",
    
    # 10 - Military
    10: "Military", 101: "Military", 102: "Military", 103: "Military"
}

mother_occupation_mapping = {
    # 0 - Students & Others
    0: "Student/Other", 90: "Student/Other", 99: "Student/Other",
    
    # 1 - Management
    1: "Management",
    
    # 2 - High-Level Specialists
    2: "High-Level Specialists", 122: "High-Level Specialists", 
    123: "High-Level Specialists", 125: "High-Level Specialists",
    
    # 3 - Intermediate / Tech
    3: "Intermediate/Tech", 131: "Intermediate/Tech", 
    132: "Intermediate/Tech", 134: "Intermediate/Tech",
    
    # 4 - Administrative Support
    4: "Administrative", 141: "Administrative", 
    143: "Administrative", 144: "Administrative",
    
    # 5 - Service & Sales
    5: "Service/Sales", 151: "Service/Sales", 
    152: "Service/Sales", 153: "Service/Sales",
    
    # 6 - Skilled Agri & Fishery
    6: "Skilled Agri/Fishery",
    
    # 7 - Skilled Trades / Crafts
    7: "Skilled Trades", 171: "Skilled Trades", 
    173: "Skilled Trades", 175: "Skilled Trades",
    
    # 8 - Operators & Assembly
    8: "Operators/Assembly",
    
    # 9 - Unskilled / Elementary
    9: "Unskilled/Elementary", 191: "Unskilled/Elementary", 192: "Unskilled/Elementary", 
    193: "Unskilled/Elementary", 194: "Unskilled/Elementary",
    
    # 10 - Military
    10: "Military"
}

# Comprehensive mapping for the 28-code list
parent_education_mapping = {
    # 0 - Low / No Schooling (New category for this list)
    35: "Low/No_Schooling", 36: "Low/No_Schooling",
    
    # 1 - Basic Education (Up to 9th Grade / 3rd Cycle)
    11: "Basic_Education", 19: "Basic_Education", 26: "Basic_Education", 
    27: "Basic_Education", 29: "Basic_Education", 30: "Basic_Education", 
    37: "Basic_Education", 38: "Basic_Education",
    
    # 2 - Secondary Education (12th year or High School)
    1: "Secondary", 9: "Secondary", 10: "Secondary", 
    12: "Secondary", 14: "Secondary", 18: "Secondary",
    
    # 3 - Technical / Specialized / Post-Secondary
    22: "Technical/Specialized", 39: "Technical/Specialized", 
    41: "Technical/Specialized", 42: "Technical/Specialized",
    
    # 4 - Undergraduate / Higher Ed (Bachelor's / 1st Cycle)
    2: "Undergraduate", 3: "Undergraduate", 6: "Undergraduate", 40: "Undergraduate",
    
    # 5 - Postgraduate (Master's / Doctorate / 2nd-3rd Cycles)
    4: "Postgraduate", 5: "Postgraduate", 43: "Postgraduate", 44: "Postgraduate",
    
    # Unknown
    34: np.nan
}

previous_education_mapping = {
    # 1 - Basic Education (Up to 9th Grade)
    19: "Basic_Education", 38: "Basic_Education",
    
    # 2 - Secondary Education / High School Incomplete
    1: "Secondary", 9: "Secondary", 10: "Secondary", 
    12: "Secondary", 14: "Secondary", 15: "Secondary",
    
    # 3 - Technical / Post-Secondary Non-Tertiary
    39: "Technical/Specialized", 42: "Technical/Specialized",
    
    # 4 - Undergraduate / Higher Ed (Bachelor's / 1st Cycle)
    2: "Undergraduate", 3: "Undergraduate", 6: "Undergraduate", 40: "Undergraduate",
    
    # 5 - Postgraduate (Master's / Doctorate)
    4: "Postgraduate", 5: "Postgraduate", 43: "Postgraduate"
}

continent_mapping = {
    # Europe
    1: "Portugese", 2: "Europe", 6: "Europe", 11: "Europe", 13: "Europe", 
    14: "Europe", 17: "Europe", 62: "Europe", 100: "Europe", 103: "Europe", 105: "Europe",
    
    # Africa (PALOP countries mostly)
    21: "Africa", 22: "Africa", 24: "Africa", 25: "Africa", 26: "Africa",
    
    # South/Central America
    41: "South_America", 101: "South_America", 108: "South_America", 109: "South_America",
    
    # Asia/Middle East
    32: "Asia"
}

df["Mother's qualification"] = df["Mother's qualification"].map(parent_education_mapping)
df["Father's qualification"] = df["Father's qualification"].map(parent_education_mapping)
df["Previous qualification"] = df["Previous qualification"].map(previous_education_mapping)
df["Mother's occupation"] = df["Mother's occupation"].map(mother_occupation_mapping)
df["Father's occupation"] = df["Father's occupation"].map(father_occupation_mapping)
df["Nacionality"] = df["Nacionality"].map(continent_mapping)

df = df.dropna()
print(df.isna().sum())
df.to_csv('data/processed/data_mapped.csv', index=False, sep=';')
