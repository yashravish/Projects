import React, { useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { Product } from '../types/database';
import { useCartStore } from '../store/cartStore';
import { Plus, Package2 } from 'lucide-react';
import toast from 'react-hot-toast';

export default function Products() {
  const [products, setProducts] = useState<Product[]>([]);
  const addToCart = useCartStore((state) => state.addItem);

  useEffect(() => {
    fetchProducts();
  }, []);

  async function fetchProducts() {
    const { data, error } = await supabase
      .from('products')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) {
      toast.error('Failed to fetch products');
      return;
    }

    setProducts(data || []);
  }

  function handleAddToCart(product: Product) {
    addToCart(product, 1);
    toast.success('Added to cart');
  }

  return (
    <div className="animate-fade-in">
      <h1 className="text-3xl font-bold mb-8 text-gray-800">
        Available Products
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {products.map((product) => (
          <div key={product.id} className="card p-6 group">
            <div className="flex items-center justify-center h-40 mb-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg">
              <Package2 className="w-16 h-16 text-blue-500 group-hover:scale-110 transition-transform duration-200" />
            </div>
            <h2 className="text-xl font-semibold mb-2 text-gray-800">{product.name}</h2>
            <p className="text-gray-600 mb-4 line-clamp-2">{product.description}</p>
            <div className="flex items-center justify-between mt-auto">
              <div>
                <span className="text-2xl font-bold text-gray-900">
                  ${product.price}
                </span>
                <div className="text-sm text-gray-500 mt-1">
                  {product.stock} in stock
                </div>
              </div>
              <button
                onClick={() => handleAddToCart(product)}
                disabled={product.stock <= 0}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Plus className="w-4 h-4" />
                Add to Cart
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}