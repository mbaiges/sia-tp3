import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

finished_avg_error = False
finished_genetic_diversity = False
finished_best_ind_stats = False

iteration = 0

def plot_avg_error(q):

    global finished_avg_error

    finished_avg_error = False

    iters_val = []
    # error
    avg_error = []

    fig = plt.figure()

    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)

    def animate(i): 
        global finished_avg_error, iteration

        if finished_avg_error:
            return

        avg_error_data = q.get()

        if avg_error_data == "STOP":
            finished_avg_error = True
            return

        iters_val.append(iteration)
        avg_error.append(avg_error_data['mean_error'])

        ax1.clear()

        plt.xlabel("Iterations")
        plt.ylabel("Avg Error")
        plt.title("Avg Error Real-Time")

        l1 = ax1.plot(iters_val, avg_error, 'r-')

        iteration += 1
        
    ani = animation.FuncAnimation(fig, animate, interval=0.02) 
    plt.show()

    return