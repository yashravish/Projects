// src/components/ProgressChart.js
import React from 'react';
import { View, Text } from 'react-native';

export default function ProgressChart() {
  return (
    <View className="p-4">
      <Text className="text-xl font-bold mb-4">Your Progress</Text>
      <Text>Visualize your mood tracking and journal entries over time.</Text>
      {/* Integrate a chart library or custom SVG here */}
    </View>
  );
}
