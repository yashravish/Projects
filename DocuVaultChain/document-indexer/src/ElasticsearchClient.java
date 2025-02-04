// document-indexer/src/main/java/com/docuvaulchain/ElasticsearchClient.java
package com.docuvaulchain;

import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;

public class ElasticsearchClient {
    public static RestHighLevelClient createClient() {
        return new RestHighLevelClient(
            RestClient.builder(
                // Replace with your Elasticsearch host and port
                new org.apache.http.HttpHost("localhost", 9200, "http")
            )
        );
    }
}
