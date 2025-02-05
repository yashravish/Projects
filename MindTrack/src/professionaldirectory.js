// src/components/ProfessionalDirectory.js
import React from 'react';
import { View, Text } from 'react-native';

export default function ProfessionalDirectory() {
  return (
    <View className="p-4">
      <Text className="text-xl font-bold mb-4">Professional Directory</Text>
      <Text>Browse mental health professionals available for consultation.</Text>
      {/* Render a list of professionals here */}
    </View>
  );
}
