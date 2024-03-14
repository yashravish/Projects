public class Cafeteria {
    private Table[] tables;

    public Cafeteria(int numTables, int seatsPerTable) {
        this.tables = new Table[numTables];
        
        for (int i = 0; i < numTables; i++) {
            tables[i] = new Table(seatsPerTable);
        }
    }
     public void takeSeats(int tableIndex, int numSeats) {
        if (tableIndex >= 0 && tableIndex < tables.length) {
            Table table = tables[tableIndex];
            table.takeSeats(numSeats);
        } else {
            System.out.println("Invalid table index.");
        }
    }
    public void releaseSeats(int tableIndex, int numSeats) {
        if (tableIndex >= 0 && tableIndex < tables.length) {
            Table table = tables[tableIndex];
            table.releaseSeats(numSeats);
        } else {
            System.out.println("Invalid table index.");
        }
    }
    public Table[] getTables() {
        return this.tables;
    }
}




