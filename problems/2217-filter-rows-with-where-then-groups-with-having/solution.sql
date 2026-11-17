-- Repeat buyers by completed orders
SELECT customer_id, count(*) as completed_orders, sum(amount) AS total_spent
from orders
where status = 'completed'
group by customer_id
having count(*) >= 2 and sum(amount)>=100

