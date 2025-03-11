import { createClient } from '@supabase/supabase-js';
import { Database } from '../types/supabase';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey);

export async function decrementStock(productId: string, quantity: number) {
  const { data: product, error: fetchError } = await supabase
    .from('products')
    .select('stock')
    .eq('id', productId)
    .single();

  if (fetchError || !product) {
    throw new Error('Failed to fetch product stock');
  }

  if (product.stock < quantity) {
    throw new Error('Insufficient stock');
  }

  const { error: updateError } = await supabase
    .from('products')
    .update({ stock: product.stock - quantity })
    .eq('id', productId);

  if (updateError) {
    throw new Error('Failed to update stock');
  }
}