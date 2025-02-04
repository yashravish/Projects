// document-indexer/src/main/java/com/docuvaulchain/DocumentIndexer.java
package com.docuvaulchain;

import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.client.RestHighLevelClient;
import org.elasticsearch.common.xcontent.XContentType;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class DocumentIndexer {
    private RestHighLevelClient client;

    public DocumentIndexer(RestHighLevelClient client) {
        this.client = client;
    }

    public void indexDocument(String indexName, String documentId, String content) throws IOException {
        Map<String, Object> jsonMap = new HashMap<>();
        jsonMap.put("content", content);
        jsonMap.put("documentId", documentId);
        IndexRequest request = new IndexRequest(indexName).id(documentId)
                .source(jsonMap, XContentType.JSON);
        IndexResponse response = client.index(request, RequestOptions.DEFAULT);
        System.out.println("Document indexed with id: " + response.getId());
    }
}
