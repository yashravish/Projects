public class Table {
    private int seatsPerTable;
    private Seat[] seats;
    private int takenSeats;

        public Table(int seatsPerTable) {
        this.seatsPerTable = seatsPerTable;
        this.seats = new Seat[seatsPerTable];
        this.takenSeats = 0;
        
        for (int i = 0; i < seatsPerTable; i++) {
            seats[i] = new Seat();
        }
    }
        public int getSeatsPerTable() {
        return seatsPerTable;
    }

    public int getTakenSeats() {
    return takenSeats;
    }

    public int getAvailableSeats() {
        return seatsPerTable - takenSeats;
    } 
    public void takeSeats(int numSeats) {
    if (numSeats <= getAvailableSeats()) {
        int seatsTaken = 0;
        for (int i = 0; i < seats.length; i++) {
            Seat seat = seats[i];
            if (!seat.isTaken()) {
                seat.take();
                seatsTaken++;
                if (seatsTaken == numSeats) {
                    break;
                }
            }
        }
        takenSeats += seatsTaken;
        System.out.println(numSeats + " seats taken. " + getAvailableSeats() + " seats at this table remain.");
    } else {
        System.out.println("Not enough available seats.");
    }
} 
public void releaseSeats(int numSeats) {
    if (numSeats <= takenSeats) {
        int seatsReleased = 0;
        for (int i = 0; i < seats.length; i++) {
            Seat seat = seats[i];
            if (seat.isTaken()) {
                seat.release();
                seatsReleased++;
                if (seatsReleased == numSeats) {
                    break;
                }
            }
        }
        takenSeats -= seatsReleased;
        System.out.println(numSeats + " seat(s) released. " + getAvailableSeats() + " seat(s) remaining.");
    } else {
        System.out.println("Invalid number of seats to release.");
    }
}
    }

