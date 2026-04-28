.PHONY: up down generate-data run-consumer run-pipeline test restart-airflow shell-postgres

up:
	docker-compose up -d

down:
	docker-compose down -v

generate-data:
	docker exec -it $$(docker-compose ps -q airflow-webserver) python /opt/airflow/src/producers/market_event_producer.py --records 100000

run-consumer:
	docker exec -it $$(docker-compose ps -q airflow-webserver) python /opt/airflow/src/consumers/minio_writer.py

run-pipeline:
	docker exec -it $$(docker-compose ps -q airflow-webserver) airflow dags trigger batch_ingest_market_data

shell-postgres:
	docker exec -it $$(docker-compose ps -q postgres) psql -U postgres -d data_warehouse
