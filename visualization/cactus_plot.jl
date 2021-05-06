using CSV
using PGFPlots 
using DataFrames

function add_cactus!(plot, times, label, style)
    sorted = sort(times)
    accumulated = accumulate(+, sorted)
    push!(plot, Plots.Linear(1:length(times), accumulated, legendentry=label, markSize=2.0, style=style))
end

function cactus_plot!(plot, times, labels, styles)
    # add a line for each column in the table
    for i = 1:length(times)
        add_cactus!(plot, times[i], labels[i], styles[i])
    end
end

function clean_times!(times, time_cutoff)
    for time_vec in times
         filter!(x -> x<=time_cutoff, time_vec)
    end
end

function clean_and_cactus_plot(times, labels, styles, max_time, output_file; title="Cactus Plot")
    plot = Axis(style="black, width=19cm, height=12cm", xlabel="Solved Instances", ylabel="Time (s)", title=title)
    #plot.legendStyle = "anchor = north west"
    plot.legendPos = "north west"
    clean_times!(times, max_time)
    println("cleaned times: ", times)
    cactus_plot!(plot, times, labels, styles)

    # Now save the figure
    save(output_file*".pdf", plot)
    save(output_file*".tex", plot)
end
