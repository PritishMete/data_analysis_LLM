# Myntra product catalogue

## What this dataset represents

This is shopping-site product data from Myntra. Each row represents a product listing that can be used to study product assortment, pricing, discounts, ratings, and available sizes.

## Shape

- Rows: 24,324 product listings
- Columns: 8
- Granularity: one row per product listing

## Columns

| Column | Description |
| --- | --- |
| `product_name` | Name or title of the product listing |
| `brand_name` | Brand associated with the product |
| `rating` | Customer rating recorded for the listing |
| `rating_count` | Number of ratings recorded for the listing |
| `marked_price` | Listed price before discount |
| `discounted_price` | Current or discounted selling price |
| `sizes` | Available sizes stored as a comma-separated value |
| `product_link` | Product path or link identifier |

## Suitable analysis

This dataset supports product and brand counts, price comparisons, discount calculations, rating analysis, popular-product ranking, size availability analysis, and questions such as which brands have the most listings or which products offer the largest price reductions.

The source file was an Excel workbook in the public `JayanGupta/Notes` repository and is provided here as `myntra-dataset.csv` for consistent use by the learner.
