/*
  # Add admin policies for products table

  1. Security
    - Add policy for admins to perform all operations on products table
    - Ensure admins can manage products (create, read, update, delete)

  Note: This maintains existing policies while adding admin-specific access
*/

CREATE POLICY "Admins can manage products"
  ON products
  FOR ALL
  TO authenticated
  USING (auth.jwt() ->> 'role' = 'admin')
  WITH CHECK (auth.jwt() ->> 'role' = 'admin');