// src/components/CommunityGroup.js
import React from 'react';
import { View, Text } from 'react-native';

export default function CommunityGroup() {
  return (
    <View className="p-4">
      <Text className="text-xl font-bold mb-4">Community Support</Text>
      <Text>Join the conversation with others who care about mental wellness.</Text>
      {/* Here you can integrate a chat or forum component */}
    </View>
  );
}
