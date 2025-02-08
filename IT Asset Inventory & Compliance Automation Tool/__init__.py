# cloud-functions/license-checker/__init__.py
def main(timer: func.TimerRequest) -> None:
    expiring_soon = get_expiring_licenses(days=30)
    for license in expiring_soon:
        send_alert(
            f"License {license.product_name} expires {license.expiration_date}",
            recipients=license.owners
        )
    update_license_statuses()