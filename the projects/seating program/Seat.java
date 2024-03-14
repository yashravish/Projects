public class Seat{
    private boolean taken; 
    
    public Seat() {
        this.taken = false;
    }

    public boolean isTaken() {
        return taken;
    }

    public void take() {
        this.taken = true;
    }

    public void release() {
        this.taken = false;
    }
}