# infrastructure/azure/main.tf
resource "azurerm_key_vault" "asset_vault" {
  name                = "asset-mgmt-secrets"
  resource_group_name = azurerm_resource_group.main.name
  sku_name            = "standard"
}

resource "azurerm_role_assignment" "rbac" {
  scope                = azurerm_key_vault.asset_vault.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.main.principal_id
}