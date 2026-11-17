-- Every product with its number of reviews
SELECT p.product_id, p.name, COUNT(r.review_id) AS review_count
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.name;