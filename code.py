
# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from google.colab import files


# ============================================================
# 1. WEB SCRAPING
# ============================================================

base_url = "https://books.toscrape.com/catalogue/page-{}.html"
books = []

print("Starting web scraping...")

for page in range(1, 13):

    url = base_url.format(page)

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for book in soup.select("article.product_pod"):

            title = book.h3.a["title"]
            price = book.select_one("p.price_color").text.strip()
            rating = book.p["class"][1]
            availability = book.select_one("p.instock").text.strip()

            books.append({
                "Title": title,
                "Price": price,
                "Rating": rating,
                "Availability": availability
            })

        print(f"Page {page} completed — total books: {len(books)}")
        time.sleep(0.5)

    except requests.RequestException as error:
        print(f"Error scraping page {page}: {error}")


# Check whether any data was collected

if not books:
    raise RuntimeError(
        "No books were scraped. Check the internet connection and try again."
    )


# Create raw DataFrame

df_raw = pd.DataFrame(books)

print("\nRAW DATASET")
print("Shape:", df_raw.shape)
print(df_raw.head())


# ============================================================
# 2. SAVE CSV 1: RAW DATA
# ============================================================

raw_file = "raw_books_data.csv"
df_raw.to_csv(raw_file, index=False)

print(f"\nSaved: {raw_file}")


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

df = df_raw.copy()

# Convert price strings to numeric values

df["Price"] = (
    df["Price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)


# Convert star ratings into numbers

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["Rating"] = df["Rating"].map(rating_map)


# Extract stock quantity from availability text

df["Availability"] = df["Availability"].str.extract(
    r"(\d+)"
)[0]

df["Availability"] = pd.to_numeric(
    df["Availability"], errors="coerce"
)


# Display missing values

print("\nMissing values before handling:")
print(df.isnull().sum())


# Remove rows missing essential fields

df = df.dropna(subset=["Title", "Price", "Rating"])


# Fill missing stock quantities with 0

df["Availability"] = df["Availability"].fillna(0)


# Remove duplicate title-price combinations

print(
    "\nDuplicate records before removal:",
    df.duplicated(subset=["Title", "Price"]).sum()
)

df = df.drop_duplicates(
    subset=["Title", "Price"]
).copy()

print(
    "Duplicate records after removal:",
    df.duplicated(subset=["Title", "Price"]).sum()
)


# Display preprocessed dataset

print("\nPREPROCESSED DATASET")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


# ============================================================
# 4. SAVE CSV 2: PREPROCESSED DATA
# ============================================================

preprocessed_file = "preprocessed_books_data.csv"
df.to_csv(preprocessed_file, index=False)

print(f"\nSaved: {preprocessed_file}")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

df_featured = df.copy()

df_featured["Price_Range"] = pd.cut(
    df_featured["Price"],
    bins=[0, 20, 40, 100],
    labels=["Low", "Mid", "High"],
    include_lowest=True
)

print("\nFEATURE-ENGINEERED DATASET")
print("Shape:", df_featured.shape)
print("Columns:", df_featured.columns.tolist())
print(df_featured.head())

print("\nPrice Range Distribution:")
print(df_featured["Price_Range"].value_counts().sort_index())


# ============================================================
# 6. SAVE CSV 3: FEATURE-ENGINEERED DATA
# ============================================================

feature_file = "feature_engineered_books_data.csv"

df_featured.to_csv(feature_file, index=False)

print(f"\nSaved: {feature_file}")


# ============================================================
# 7. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n========== BASIC STATISTICS ==========")
print(df.describe())

print("\nAverage Price by Rating:")
print(df.groupby("Rating")["Price"].mean().round(2))

print("\nPrice Range Distribution:")
print(df_featured["Price_Range"].value_counts().sort_index())


# ============================================================
# 8. DATA VISUALISATIONS
# ============================================================

plt.figure(figsize=(14, 10))


# 1. Price Distribution

plt.subplot(2, 2, 1)

sns.histplot(
    df["Price"],
    bins=20,
    kde=True,
    color="steelblue"
)

plt.title("Price Distribution of Books", fontsize=13)
plt.xlabel("Price (£)")
plt.ylabel("Count")


# 2. Average Price by Star Rating

plt.subplot(2, 2, 2)

avg_price = df.groupby("Rating")["Price"].mean().sort_index()

avg_price.plot(
    kind="bar",
    color="teal",
    edgecolor="black"
)

plt.title("Average Price by Star Rating", fontsize=13)
plt.ylabel("Average Price (£)")
plt.xlabel("Rating")
plt.xticks(rotation=0)


# 3. Average Price by Rating - Line Chart

plt.subplot(2, 2, 3)

avg_price.plot(
    kind="line",
    marker="o",
    color="darkorange",
    linewidth=2.5,
    markersize=9
)

plt.title("Average Price by Star Rating", fontsize=13)
plt.ylabel("Average Price (£)")
plt.xlabel("Star Rating")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks([1, 2, 3, 4, 5])


# 4. Books by Price Range

plt.subplot(2, 2, 4)

df_featured["Price_Range"].value_counts().sort_index().plot(
    kind="bar",
    color="coral",
    edgecolor="black"
)

plt.title("Books by Price Range", fontsize=13)
plt.ylabel("Number of Books")
plt.xlabel("Price Range")
plt.xticks(rotation=0)


plt.tight_layout()
plt.show()


# ============================================================
# 9. KEY INSIGHTS
# ============================================================

print("\n========== KEY INSIGHTS ==========")

print(f"Books scraped before preprocessing: {len(df_raw)}")
print(f"Books after preprocessing: {len(df)}")
print(f"Average price: £{df['Price'].mean():.2f}")
print(f"Average rating: {df['Rating'].mean():.2f} stars")

print(
    f"Most common price range: "
    f"{df_featured['Price_Range'].mode().iloc[0]}"
)

print(f"Highest priced book: £{df['Price'].max():.2f}")
print(f"Lowest priced book: £{df['Price'].min():.2f}")


# ============================================================
# 10. CSV FILES READY FOR DOWNLOAD
# ============================================================

print("\n========== CSV FILES READY ==========")

print(raw_file)
print(preprocessed_file)
print(feature_file)

print("\nStarting downloads...")

files.download(raw_file)
time.sleep(1)

files.download(preprocessed_file)
time.sleep(1)

files.download(feature_file)

print("\nAll three CSV files have been sent for download.")
