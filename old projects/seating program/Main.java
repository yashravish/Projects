import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Cafeteria cafeteria = new Cafeteria(2, 4);
        Scanner scanner = new Scanner(System.in);
        
        int input = 0;
        while (input != 4) {
            System.out.println("Type 1 to take a seat. Type 2 to leave a seat. Type 3 for an update. Type 4 to exit.");
            input = scanner.nextInt();
    
            if (input == 1) {
                boolean seatTaken = false;
                Table[] tables = cafeteria.getTables();
    
                for (int tableIndex = 0; tableIndex < tables.length; tableIndex++) {
                    Table table = tables[tableIndex];
    
                    if (table.getAvailableSeats() > 0) {
                        table.takeSeats(1);
                        seatTaken = true;
                        break;
                    }
                }
    
                if (!seatTaken) {
                    System.out.println("No available seats.");
                }
            } else if (input == 2) {
                boolean seatReleased = false;
                Table[] tables = cafeteria.getTables();
    
                for (int tableIndex = 0; tableIndex < tables.length; tableIndex++) {
                    Table table = tables[tableIndex];
    
                    if (table.getTakenSeats() > 0) {
                        table.releaseSeats(1);
                        seatReleased = true;
                        break;
                    }
                }
    
                if (!seatReleased) {
                    System.out.println("No seats to release.");
                }
            } else if (input == 3) {
                Table[] tables = cafeteria.getTables();
                int availableTables = 0;
                int availableSeats = 0;
                int totalSeats = 0;
    
                for (int tableIndex = 0; tableIndex < tables.length; tableIndex++) {
                    Table table = tables[tableIndex];
                    availableTables++;
                    availableSeats += table.getAvailableSeats();
                    totalSeats += table.getSeatsPerTable();
                }
    
                System.out.println("Available tables: " + availableTables);
                System.out.println("Available seats: " + availableSeats);
                System.out.println("Total seats in the cafeteria: " + totalSeats);
            } else if (input == 4) {
                System.out.println("Exiting the program.");
            } else {
                System.out.println("Invalid input: " + input);
            }
        }
    }
}
