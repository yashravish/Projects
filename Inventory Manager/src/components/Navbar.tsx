import React from 'react';
import { Link } from 'react-router-dom';
import { ShoppingCart, Package, LayoutDashboard, Store } from 'lucide-react';
import { useCartStore } from '../store/cartStore';

export default function Navbar() {
  const cartItems = useCartStore((state) => state.items);

  return (
    <nav className="bg-white shadow-lg sticky top-0 z-50 backdrop-blur-lg bg-white/80">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2 text-xl font-bold text-gray-800">
            <Store className="w-6 h-6 text-blue-500" />
            <span>Inventory Manager</span>
          </Link>
          
          <div className="flex items-center space-x-6">
            <Link
              to="/"
              className="text-gray-600 hover:text-blue-500 flex items-center space-x-1 transition-colors duration-200"
            >
              <Package className="w-5 h-5" />
              <span>Products</span>
            </Link>
            
            <Link
              to="/orders"
              className="text-gray-600 hover:text-blue-500 flex items-center space-x-1 transition-colors duration-200"
            >
              <LayoutDashboard className="w-5 h-5" />
              <span>Orders</span>
            </Link>
            
            <Link
              to="/cart"
              className="text-gray-600 hover:text-blue-500 flex items-center space-x-1 transition-colors duration-200 relative"
            >
              <ShoppingCart className="w-5 h-5" />
              <span>Cart</span>
              {cartItems.length > 0 && (
                <span className="absolute -top-2 -right-2 bg-blue-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold animate-fade-in">
                  {cartItems.length}
                </span>
              )}
            </Link>
            
            <Link
              to="/admin"
              className="text-gray-600 hover:text-blue-500 flex items-center space-x-1 transition-colors duration-200"
            >
              <span>Admin</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}