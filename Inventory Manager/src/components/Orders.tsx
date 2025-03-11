import React, { useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { Order, OrderItem, Product } from '../types/database';
import toast from 'react-hot-toast';

interface OrderWithItems extends Order {
  order_items: (OrderItem & { products: Product })[];
}

export default function Orders() {
  const [orders, setOrders] = useState<OrderWithItems[]>([]);

  useEffect(() => {
    fetchOrders();
  }, []);

  async function fetchOrders() {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      toast.error('Please sign in to view orders');
      return;
    }

    const { data, error } = await supabase
      .from('orders')
      .select(`
        *,
        order_items (
          *,
          products (*)
        )
      `)
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });

    if (error) {
      toast.error('Failed to fetch orders');
      return;
    }

    setOrders(data || []);
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Your Orders</h1>
      <div className="space-y-6">
        {orders.map((order) => (
          <div key={order.id} className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <span className="text-sm text-gray-500">Order placed</span>
                <p className="font-semibold">
                  {new Date(order.created_at).toLocaleDateString()}
                </p>
              </div>
              <div>
                <span className="text-sm text-gray-500">Total</span>
                <p className="font-semibold">${order.total_amount}</p>
              </div>
              <div>
                <span className="text-sm text-gray-500">Status</span>
                <p className="font-semibold capitalize">{order.status}</p>
              </div>
            </div>
            <div className="border-t pt-4">
              {order.order_items.map((item) => (
                <div
                  key={item.id}
                  className="flex justify-between items-center py-2"
                >
                  <div>
                    <p className="font-semibold">{item.products.name}</p>
                    <p className="text-sm text-gray-500">
                      Quantity: {item.quantity}
                    </p>
                  </div>
                  <p className="font-semibold">
                    ${(item.unit_price * item.quantity).toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}