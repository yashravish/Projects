// src/App.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import AppNavigator from './navigation/AppNavigator';
import { WellnessProvider } from './context/WellnessContext';

export default function App() {
  return (
    <WellnessProvider>
      <NavigationContainer>
        <AppNavigator />
      </NavigationContainer>
    </WellnessProvider>
  );
}
