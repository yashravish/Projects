const { ApolloServer, gql } = require('apollo-server');
const fs = require('fs');
const path = require('path');

// Load the GraphQL schema from the file.
const schemaPath = path.join(__dirname, '../schema.graphql');
const typeDefs = gql(fs.readFileSync(schemaPath, { encoding: 'utf-8' }));

// Sample resolvers – in production, aggregate real data from Java, Python, and .NET services.
const resolvers = {
    Query: {
        getPreApproval: async (_, { applicantId }) => {
            // Simulated values:
            const ficoScore = 720.0;         // Dummy FICO score.
            const cashFlowScore = 1200.0;     // Dummy average cash flow.
            const preApproved = ficoScore >= 700 && cashFlowScore >= 1000;
            return {
                applicantId,
                ficoScore,
                cashFlowScore,
                preApproved
            };
        }
    }
};

const server = new ApolloServer({ typeDefs, resolvers });

server.listen({ port: 5000 }).then(({ url }) => {
    console.log(`GraphQL API running at ${url}`);
});
