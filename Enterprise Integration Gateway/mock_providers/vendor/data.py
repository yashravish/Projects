"""
Realistic Vendor XML mock data.

Uses PascalCase XML tags (different from CRM and different from internal schema)
to demonstrate XML-to-normalized-schema transformation.

Intentionally includes one MALFORMED order record (missing OrderId, invalid TotalAmount)
to demonstrate dead-letter / failed_records handling.
"""

ORDERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<OrderFeed>
    <!-- Valid orders -->
    <Order>
        <OrderId>VND-ORD-2001</OrderId>
        <ExternalCustomerId>CRM-CUST-001</ExternalCustomerId>
        <OrderDate>2024-01-12T08:30:00Z</OrderDate>
        <Status>confirmed</Status>
        <Currency>USD</Currency>
        <TotalAmount>645.00</TotalAmount>
        <Notes>Vendor-fulfilled portion of enterprise contract</Notes>
    </Order>
    <Order>
        <OrderId>VND-ORD-2002</OrderId>
        <ExternalCustomerId>CRM-CUST-002</ExternalCustomerId>
        <OrderDate>2024-01-28T11:00:00Z</OrderDate>
        <Status>shipped</Status>
        <Currency>USD</Currency>
        <TotalAmount>1120.50</TotalAmount>
        <Notes></Notes>
    </Order>
    <Order>
        <OrderId>VND-ORD-2003</OrderId>
        <ExternalCustomerId>CRM-CUST-003</ExternalCustomerId>
        <OrderDate>2024-02-10T14:15:00Z</OrderDate>
        <Status>delivered</Status>
        <Currency>USD</Currency>
        <TotalAmount>299.99</TotalAmount>
        <Notes>Standard delivery</Notes>
    </Order>
    <!-- MALFORMED record: missing OrderId and invalid TotalAmount — should go to failed_records -->
    <Order>
        <OrderId></OrderId>
        <ExternalCustomerId>CRM-CUST-UNKNOWN</ExternalCustomerId>
        <OrderDate>2024-02-14T00:00:00Z</OrderDate>
        <Status>unknown</Status>
        <Currency>USD</Currency>
        <TotalAmount>INVALID_AMOUNT</TotalAmount>
        <Notes>This record is intentionally malformed for testing</Notes>
    </Order>
</OrderFeed>
"""

SHIPMENTS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ShipmentFeed>
    <Shipment>
        <ShipmentId>VND-SHIP-3001</ShipmentId>
        <VendorOrderId>VND-ORD-2001</VendorOrderId>
        <TrackingNumber>1Z999AA10123456784</TrackingNumber>
        <Carrier>UPS</Carrier>
        <Status>delivered</Status>
        <EstimatedDelivery>2024-01-18T00:00:00Z</EstimatedDelivery>
        <ActualDelivery>2024-01-17T14:32:00Z</ActualDelivery>
        <WeightKg>3.200</WeightKg>
    </Shipment>
    <Shipment>
        <ShipmentId>VND-SHIP-3002</ShipmentId>
        <VendorOrderId>VND-ORD-2002</VendorOrderId>
        <TrackingNumber>9400111899223397123456</TrackingNumber>
        <Carrier>USPS</Carrier>
        <Status>in_transit</Status>
        <EstimatedDelivery>2024-02-05T00:00:00Z</EstimatedDelivery>
        <ActualDelivery></ActualDelivery>
        <WeightKg>1.750</WeightKg>
    </Shipment>
    <Shipment>
        <ShipmentId>VND-SHIP-3003</ShipmentId>
        <VendorOrderId>VND-ORD-2003</VendorOrderId>
        <TrackingNumber>JD014600004987654321</TrackingNumber>
        <Carrier>FedEx</Carrier>
        <Status>delivered</Status>
        <EstimatedDelivery>2024-02-17T00:00:00Z</EstimatedDelivery>
        <ActualDelivery>2024-02-16T10:00:00Z</ActualDelivery>
        <WeightKg>0.850</WeightKg>
    </Shipment>
    <Shipment>
        <ShipmentId>VND-SHIP-3004</ShipmentId>
        <VendorOrderId>VND-ORD-2002</VendorOrderId>
        <TrackingNumber>7489044600499267537</TrackingNumber>
        <Carrier>DHL</Carrier>
        <Status>out_for_delivery</Status>
        <EstimatedDelivery>2024-02-06T00:00:00Z</EstimatedDelivery>
        <ActualDelivery></ActualDelivery>
        <WeightKg>2.100</WeightKg>
    </Shipment>
</ShipmentFeed>
"""
