//~~~~~~~~~~~~~~~~~~~~~~~~~~~  JOB SEQUENCING ~~~~~~~~~~~~~~~~~~~~~~

import java.util.Arrays;

class JOB{
    String jobID ;
    int dead ;
    int prof ;

    public JOB (String jobID, int dead ,int prof ){
        this.jobID = jobID;
        this.dead = dead;
        this.prof = prof;
    }

}

class JobSequence{

    public static void jobSequencing (JOB [] jobs , int MaxDeadline){
        String [] result = new String [MaxDeadline];
        Arrays.fill(result , null);
        int totalProfit = 0;

        Arrays.sort (jobs , (a,b)-> b.prof - a.prof);

        for(JOB job : jobs){
            for( int slot = Math.min(MaxDeadline-1 , job.dead-1); slot>=0 ; slot-- ){
                if (result[slot] == null) {
                    result[slot] = job.jobID;
                    totalProfit += job.prof;
                    break;
                }
            }

        }

        System.out.print("job sequence : ");
        for(String jo: result ){
            System.out.print(jo + " ");
        }

        System.out.println("Total profit - " + totalProfit);

    }

    public static void main(String[] args) {
        
        JOB [] jobs = {
            new JOB("j1" , 2 ,100),
            new JOB("j2" , 1 ,19),
            new JOB("j3" , 2 ,27),
            new JOB("j4" , 1 ,25),
            new JOB("j5" , 3 ,15),
        };

        int MaxDeadline = 3;

        jobSequencing(jobs , MaxDeadline);
    }

}



// public static void main(String[] args) {
//     Scanner scanner = new Scanner(System.in);

//     // Take number of jobs as input
//     System.out.print("Enter the number of jobs: ");
//     int n = scanner.nextInt();

//     // Initialize the jobs array
//     JOB[] jobs = new JOB[n];
//     int maxDeadline = 0;

//     // Take job details as input
//     for (int i = 0; i < n; i++) {
//         System.out.print("Enter job ID: ");
//         String jobid = scanner.next();
        
//         System.out.print("Enter job deadline: ");
//         int deadline = scanner.nextInt();
        
//         System.out.print("Enter job profit: ");
//         int profit = scanner.nextInt();

//         jobs[i] = new JOB(jobid, deadline, profit);

//         // Update maxDeadline to the highest deadline encountered
//         if (deadline > maxDeadline) {
//             maxDeadline = deadline;
//         }
//     }

//     // Call job sequencing function
//     jobsequencing(jobs, maxDeadline);

//     scanner.close();
// }