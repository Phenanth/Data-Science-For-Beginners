-- Assignment 05

-- Create queries to return the following information:

-- 1. all city names in the Cities table
SELECT DISTINCT city
FROM Cities;

-- 2. all cities in Ireland in the Cities table
SELECT DISTINCT city
FROM Cities
WHERE country = 'Ireland';

-- 3. all airport names with their city and country
SELECT DISTINCT a.name, c.city, c.country
FROM Airports a
INNER JOIN Cities c ON a.city_id = c.id;

-- 4. all airports in London, United Kingdom
SELECT DISTINCT a.id, a.name, a.code
FROM Airports a
JOIN Cities c ON a.city_id = c.id
WHERE c.city = 'London'
	AND c.country = 'United Kingdom';