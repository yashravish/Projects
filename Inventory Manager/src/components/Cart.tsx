import React from 'react';
import { useCartStore } from '../store/cartStore';
import { Trash2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { supabase, decrementStock } from '../lib/supabase';
import toast from 'react-hot-toast';

export default function Cart() {
  const { items, removeItem, clearCart, total } = useCartStore();
  const navigate = useNavigate();

  async function handleCheckout() {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user) {
        toast.error('Please sign in to complete your order');
        return;
      }

      // Start transaction
      const { data: order, error: orderError } = await supabase
        .from('orders')
        .insert({
          user_id: user.id,
          total_amount: total(),
          status: 'pending'
        })
        .select()
        .single();

      if (orderError || !order) {
        throw new Error('Failed to create order');
      }

      // Process each item
      for (const item of items) {
        try {
          await decrementStock(item.product.id, item.quantity);
          
          const { error: itemError } = await supabase
            .from('order_items')
            .insert({
              order_id: order.id,
              product_id: item.product.id,
              quantity: item.quantity,
              unit_price: item.product.price
            });

          if (itemError) {
            throw new Error('Failed to create order item');
          }
        } catch (error) {
          // Revert the order if any item fails
          await supabase
            .from('orders')
            .delete()
            .eq('id', order.id);
          
          throw error;
        }
      }

      clearCart();
      toast.success('Order placed successfully!');
      navigate('/orders');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to process order');
    }
  }

  if (items.length === 0) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold mb-4">Your cart is empty</h2>
        <button
          onClick={() => navigate('/')}
          className="text-blue-500 hover:text-blue-600"
        >
          Continue Shopping
        </button>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-8 text-gray-800">Shopping Cart</h1>
      <div className="card p-6">
        {items.map((item) => (
          <div
            key={item.product.id}
            className="flex items-center justify-between py-4 border-b last:border-0"
          >
            <div>
              <h3 className="text-lg font-semibold">{item.product.name}</h3>
              <p className="text-gray-600">
                ${item.product.price} × {item.quantity}
              </p>
            </div>
            <div className="flex items-center">
              <span className="font-bold mr-4">
                ${(item.product.price * item.quantity).toFixed(2)}
              </span>
              <button
                onClick={() => removeItem(item.product.id)}
                className="btn-danger"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))}
        <div className="mt-6 pt-6 border-t">
          <div className="flex justify-between text-xl font-bold">
            <span>Total:</span>
            <span>${total().toFixed(2)}</span>
          </div>
          <button
            onClick={handleCheckout}
            className="btn-primary w-full mt-4"
          >
            Checkout
          </button>
        </div>
      </div>
    </div>
  );
}