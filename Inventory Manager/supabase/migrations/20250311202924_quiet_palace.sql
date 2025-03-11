/*
  # Add stock management trigger

  1. Changes
    - Add trigger to prevent stock from going negative
    - Add function to validate stock levels
    
  2. Security
    - Function is security definer to ensure it runs with necessary privileges
*/

CREATE OR REPLACE FUNCTION check_stock_level()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.stock < 0 THEN
    RAISE EXCEPTION 'Stock cannot be negative';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER prevent_negative_stock
  BEFORE UPDATE OR INSERT ON products
  FOR EACH ROW
  EXECUTE FUNCTION check_stock_level();