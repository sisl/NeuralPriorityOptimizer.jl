using CSV
using PGFPlots 
using DataFrames

include(string(@__DIR__, "/cactus_plot.jl"))
data_file = string(@__DIR__, "/../results/CAS/ForWorkshop/combined.csv")
data = CSV.read(data_file, DataFrame)
max_time = 500.0

# Choose which properties you'd like to plot 
properties = [1,2,3,4]
for property in properties
    output_file = string(@__DIR__, "/plots/acas_cactus_plots/cactus_property", property)

    # Get your subset of the data
    property_data = filter(row -> row["property"] == property, data)
    
    # Put the times into a list of times, give them each labels, then plot
    # times = [property_data[:, "nnenum_time"], property_data[:, "eran_time"], property_data[:, "marabou_time"], property_data[:, "priority_time"]]
    # labels= ["NNENUM", "ERAN", "Marabou", "Priority Optimizer (ours)"]
    # styles = ["mark=+, blue", "mark=triangle, red", "mark=square, black", "mark=diamond,teal"]
    
    # Without ERAN
    times = [property_data[:, "nnenum_time"], property_data[:, "marabou_time"], property_data[:, "priority_time"]]
    labels= ["NNENUM", "Marabou", "Priority Optimizer (ours)"]
    styles = ["mark=+, blue", "mark=triangle, red", "mark=square, black"]
    clean_and_cactus_plot(times, labels, styles, max_time, output_file; title=string("Cactus Plot For Property ", property))
end
