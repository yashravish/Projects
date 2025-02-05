// src/components/EmergencyLocator.js
import React from 'react';
import { View, Text, Pressable } from 'react-native';

export default function EmergencyLocator() {
  return (
    <View className="p-4 items-center">
      <Text className="text-xl font-bold mb-4">Emergency Resources</Text>
      <Text className="mb-4">Find help immediately in case of an emergency.</Text>
      <Pressable className="bg-red-500 py-2 px-4 rounded">
        <Text className="text-white">Call Emergency Services</Text>
      </Pressable>
    </View>
  );
}
