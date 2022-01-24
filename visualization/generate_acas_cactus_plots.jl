using CSV
using PGFPlots 
using DataFrames

include(string(@__DIR__, "/cactus_plot.jl"))
data_file = string(@__DIR__, "/../results/CAS/ForWriteup/5_28_oceanside/5_28_combined_eager.csv")
data = CSV.read(data_file, DataFrame)
max_time = 300.0

# Choose which properties you'd like to plot 
properties = [1,2,3,4]
group_plot = GroupPlot(2, 2)

for property in properties
    output_file = string(@__DIR__, "/plots/acas_cactus_plots/cactus_property", property)

    # Get your subset of the data
    property_data = filter(row -> row["property"] == property, data)
    
    # Put the times into a list of times, give them each labels, then plot
    #times = [property_data[:, "nnenum_time"], property_data[:, "eran_time"], property_data[:, "marabou_time"], property_data[:, "priority_time_stopfreq1"]]
    #labels= ["NNENUM", "ERAN", "Marabou", "ZoPE (ours)"]
    #styles = ["mark=+, blue", "mark=triangle, red", "mark=square, black", "mark=diamond,teal"]
    
    #times = [property_data[:, "nnenum_time"], property_data[:, "priority_time_stopfreq1"]]
    #labels = ["NNENUM", "ZoPE (ours)"]
    #styles = ["mark=+, blue", "mark=diamond, teal"]

    # compare differnt priority optimizers
    # times = [property_data[:, "priority_time_stopfreq1"], property_data[:, "priority_time_stopfreq5"], property_data[:, "priority_time_stopfreq10"]]
    # labels = ["Priority stop freq 1", "Priority stop freq 5", "Priority stop freq 10"]
    # styles = ["mark=+, blue", "mark=triangle, red", "mark=square, black"]

    # Without ERAN
    # times = [property_data[:, "nnenum_time"], property_data[:, "marabou_time"], property_data[:, "priority_time"]]
    # labels= ["NNENUM", "Marabou", "Priority Optimizer (ours)"]
    # styles = ["mark=+, blue", "mark=triangle, red", "mark=square, black"]

    # compare eager and non-eager
    times = [property_data[:, "eager_time"], property_data[:, "noneager_time"], property_data[:, "obtuse_polytope_time"], property_data[:, "virtual_best"]]
    labels = ["eager", "noneager", "obtuse", "virtual best"]
    styles = ["mark=+, blue", "mark=triangle, red", "mark=square, black", "mark=diamond, teal"]
    
    cur_plot = clean_and_cactus_plot(times, labels, styles, max_time, output_file; title=string("Cactus Plot For Property ", property))
    push!(group_plot, cur_plot)
end

full_plot_file = string(@__DIR__, "/plots/acas_cactus_plots/compare_eager")
save(full_plot_file*".tex", group_plot)
save(full_plot_file*".pdf", group_plot)

