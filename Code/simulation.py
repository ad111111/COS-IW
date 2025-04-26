#!/usr/bin/env python

"""
BBR-S Algorithm Testing with Mininet Simulator
This script creates a network topology to evaluate the performance of BBRS
with deep buffers and bursts.
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost, OVSSwitch
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI

import time
import os
import subprocess
from multiprocessing import Process
import argparse
import threading
import numpy as np
from datetime import datetime
import json
import pandas as pd
from statistics import mean, median, stdev

DEFAULT_BW = 25  # Mbps
DEFAULT_DELAY = 50  # ms (to achieve 100ms RTT as in Vargas et al.)
DEFAULT_QUEUE_SIZE = 100  # packets
DEFAULT_BUFFER_SIZE = 10000  # (10MB buffer as in Vargas et al.)
DEFAULT_BURST_SIZE = 1000  # (1MB burst capacity as in Vargas et al.)
DEFAULT_DURATION = 60  # seconds
DEFAULT_OUTPUT_DIR = 'bbrs_results'
DEFAULT_TRIALS = 5  

class DumbbellTopo(Topo):
    """
    Creates a dumbbell topology with two senders and two receivers connected through a bottleneck link.
    The full dumbbell topology is described and illustrated in our paper.
    """
    def build(self, bw=DEFAULT_BW, delay=DEFAULT_DELAY, queue_size=DEFAULT_QUEUE_SIZE, 
              buffer_size=DEFAULT_BUFFER_SIZE, burst_size=DEFAULT_BURST_SIZE):
        
        # Two routers
        r1 = self.addSwitch('s1')
        r2 = self.addSwitch('s2')
        
        # Two clients and two servers
        sender1 = self.addHost('h1')
        sender2 = self.addHost('h2')
        receiver1 = self.addHost('h3')
        receiver2 = self.addHost('h4')
        
        # Add links between servers and router r1
        self.addLink(sender1, r1, bw=bw, delay=f'{delay}ms', max_queue_size=queue_size)
        self.addLink(sender2, r1, bw=bw, delay=f'{delay}ms', max_queue_size=queue_size)
        
        # Initialize bottleneck link between switches
        self.addLink(r1, r2, bw=bw, delay=f'{delay}ms', max_queue_size=buffer_size, burst=burst_size)
        
        # Add links between receivers and router r2
        self.addLink(r2, receiver1, bw=bw, delay=f'{delay}ms', max_queue_size=queue_size)
        self.addLink(r2, receiver2, bw=bw, delay=f'{delay}ms', max_queue_size=queue_size)

def monitor_tcp_stats(host, output_file, duration):
    """
    Monitors TCP statistics for a host, including Goodput and RTT metrics.
    Writes metrics to stats file.
    """
    stats_file = open(output_file, 'w')
    
    start_time = time.time()
    interval = 1  
    
    while time.time() - start_time < duration:
        # Get TCP stats
        tcp_stats = host.cmd("ss -tini | grep -v 'ESTAB\\|State\\|Address'")
        timestamp = time.time() - start_time
        
        stats_file.write(f"--- Time: {timestamp:.2f}s ---\n")
        stats_file.write(tcp_stats)
        stats_file.write("\n")
        
        # Get RTT stats
        rtt_info = host.cmd("ss -i | grep rtt")
        stats_file.write(rtt_info)
        stats_file.write("\n")
        
        time.sleep(interval)
    
    stats_file.close()

def measure_ping_latency(sender, receiver_ip, output_file, duration, interval=0.2):
    """
    Measures network latency with ping at an interval of 0.2s and generates RTT stats.
    Uses protocol for RTT measurement described in paper.
    """
    latency_file = open(output_file, 'w')
    latency_file.write("Time(s)\tLatency(ms)\n")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Run ping to measure current RTT
        ping_output = sender.cmd(f"ping -c 1 -W 1 {receiver_ip}")
        
        current_time = time.time() - start_time
        
        # Parse the ping output
        latency = None
        for line in ping_output.splitlines():
            if "time=" in line:
                try:
                    latency_part = line.split("time=")[1].split()[0]
                    latency = float(latency_part)
                    break
                except (IndexError, ValueError):
                    pass
        
        if latency is not None:
            latency_file.write(f"{current_time:.2f}\t{latency:.3f}\n")
        else:
            latency_file.write(f"{current_time:.2f}\tNA\n")
        
        time.sleep(interval)
    
    latency_file.close()

def start_iperf_server(host, log_file):
    """Starts an iperf server on the specified host"""

    print(f"Starting iperf server on {host.name}")
    return host.cmd(f'iperf -s > {log_file} 2>&1 &')

def start_iperf_client(host, server_ip, duration, log_file, cc_algorithm="bbrs"):
    """Starts an iperf client on the specified host"""

    print(f"Starting iperf client on {host.name} connecting to {server_ip}")
    return host.cmd(f'iperf -c {server_ip} -t {duration} -i 1 > {log_file} 2>&1 &')

def monitor_queue_size(switch, interface, output_file, duration, interval=0.2):
    """
    Monitors queue statistics on the bottleneck link (instantiated above).
    Writes output to stats file.
    """

    stats_file = open(output_file, 'w')
    stats_file.write("Time(s)\tQueueSize(packets)\tQueueDelay(ms)\n")
    
    start_time = time.time()
    
    # Get link bandwidth
    bw_info = switch.cmd(f"tc class show dev {interface}")
    bw_mbps = DEFAULT_BW
    
    # Extract bandwidth value from TCP log file
    for line in bw_info.splitlines():
        if "rate" in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "rate":
                        bw_value = parts[i+1]
                        if "Mbit" in bw_value:
                            bw_mbps = float(bw_value.split("Mbit")[0])
                        elif "Kbit" in bw_value:
                            bw_mbps = float(bw_value.split("Kbit")[0]) / 1000
                        elif "Gbit" in bw_value:
                            bw_mbps = float(bw_value.split("Gbit")[0]) * 1000
                        break
            except (IndexError, ValueError):
                pass
            break
    
    bytes_per_ms = (bw_mbps * 1000000 / 8) / 1000
    packet_size = 1500
    
    # Extract Queue stats (queue size)
    while time.time() - start_time < duration:
        cmd = f"tc -s qdisc show dev {interface}"
        queue_stats = switch.cmd(cmd)
        timestamp = time.time() - start_time
        
        queue_size = 0
        for line in queue_stats.splitlines():
            if "backlog" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "backlog":
                        try:
                            queue_size = int(parts[i+2].rstrip("p"))
                            break
                        except (IndexError, ValueError):
                            pass
                break
        
        # Calculate queuing delay
        # Delay = queue_size_in_bytes / bytes_per_ms
        queue_delay = (queue_size * packet_size) / bytes_per_ms if bytes_per_ms > 0 else 0
        
        stats_file.write(f"{timestamp:.2f}\t{queue_size}\t{queue_delay:.3f}\n")
        time.sleep(interval)
    
    stats_file.close()

def calculate_buffer_bloat_metrics(output_dir):
    """
    Calculates the following buffer bloat metrics from the experiment data:
    - RTT increase (Ratio and averages; ms)
    - Queue jitter
    - Goodput (Mbps)
    """
    results = {}
    base_rtt = None
    
    # Extract RTT results
    for filename in os.listdir(output_dir):
        if filename == "config.txt":
            with open(os.path.join(output_dir, filename), 'r') as f:
                for line in f:
                    if "RTT:" in line:
                        try:
                            base_rtt = float(line.split(":")[1].split()[0])
                            break
                        except (IndexError, ValueError):
                            pass
            break
    
    if base_rtt is None:
        base_rtt = DEFAULT_DELAY * 2
        print(f"Warning: Using default base RTT of {base_rtt}ms")
    
    # Process ping latency data for flow
    for filename in os.listdir(output_dir):
        if filename.endswith('_ping_latency.log'):
            flow_name = filename.split('_')[0]
            full_path = os.path.join(output_dir, filename)
            
            latencies = []
            timestamps = []
            
            with open(full_path, 'r') as f:
                f.readline()
                
                for line in f:
                    if not line.strip():
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    try:
                        timestamp = float(parts[0])
                        if parts[1] != "NA":
                            latency = float(parts[1])
                            latencies.append(latency)
                            timestamps.append(timestamp)
                    except ValueError:
                        continue
            
            if latencies:
                # Calculate latency metrics
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                # Get 95th percentile latency
                p95_latency = np.percentile(latencies, 95) if len(latencies) > 20 else max_latency
                
                # Calculate inflation metrics
                queue_delay = avg_latency - base_rtt
                inflation_ratio = avg_latency / base_rtt if base_rtt > 0 else 1
                
                # Calculate latency variation/jitter
                if len(latencies) > 1:
                    latency_diffs = [abs(latencies[i] - latencies[i-1]) for i in range(1, len(latencies))]
                    jitter = sum(latency_diffs) / len(latency_diffs)
                else:
                    jitter = 0
                
                # Store metrics
                results[flow_name] = {
                    'avg_rtt': avg_latency,
                    'min_rtt': min_latency,
                    'max_rtt': max_latency,
                    'p95_rtt': p95_latency, 
                    'queue_delay': queue_delay,
                    'inflation_ratio': inflation_ratio,
                    'jitter': jitter
                }
    
    # Process iperf data and compute goodput/throughput metrics
    for filename in os.listdir(output_dir):
        if filename.endswith('_iperf_client.log'):
            flow_name = filename.split('_')[0]
            full_path = os.path.join(output_dir, filename)
            
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            total_throughput = 0
            count = 0
            
            # Parse iperf output
            for line in lines:
                if 'sec' in line and 'Bytes' in line and 'bits/sec' in line and not 'SUM' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'bits/sec' in part and i > 0:
                            try:
                                value = float(parts[i-1])
                                unit = part.split('[')[0]
                                
                                if 'Kbits/sec' in unit:
                                    value /= 1000
                                elif 'Gbits/sec' in unit:
                                    value *= 1000
                                    
                                total_throughput += value
                                count += 1
                                break
                            except:
                                pass
            
            if count > 0:
                goodput = total_throughput / count
                
                if flow_name in results:
                    # Compute efficiency: throughput per unit of delay
                    if results[flow_name]['avg_rtt'] > 0:
                        efficiency = goodput / results[flow_name]['avg_rtt']
                    else:
                        efficiency = goodput
                        
                    results[flow_name]['goodput'] = goodput
                    results[flow_name]['efficiency'] = efficiency
                else:
                    results[flow_name] = {'goodput': goodput}
    
    # Parse queue monitoring data
    queue_file = os.path.join(output_dir, 'bottleneck_queue_stats.log')
    if os.path.exists(queue_file):
        queue_sizes = []
        queue_delays = []
        
        with open(queue_file, 'r') as f:
            f.readline()
            
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                try:
                    queue_size = float(parts[1])
                    queue_delay = float(parts[2])
                    
                    queue_sizes.append(queue_size)
                    queue_delays.append(queue_delay)
                except ValueError:
                    continue
        
        if queue_sizes:
            # Max and Avg queue statistics
            avg_queue_size = sum(queue_sizes) / len(queue_sizes)
            max_queue_size = max(queue_sizes)
            avg_queue_delay = sum(queue_delays) / len(queue_delays)
            max_queue_delay = max(queue_delays)
            
            # P95 and P99 queue statistics
            p95_queue_size = np.percentile(queue_sizes, 95) if len(queue_sizes) > 20 else max_queue_size
            p99_queue_size = np.percentile(queue_sizes, 99) if len(queue_sizes) > 100 else max_queue_size
            p95_queue_delay = np.percentile(queue_delays, 95) if len(queue_delays) > 20 else max_queue_delay
            
            # Store results
            results['queue_stats'] = {
                'avg_queue_size': avg_queue_size,
                'max_queue_size': max_queue_size,
                'p95_queue_size': p95_queue_size,
                'p99_queue_size': p99_queue_size,
                'avg_queue_delay': avg_queue_delay,
                'max_queue_delay': max_queue_delay,
                'p95_queue_delay': p95_queue_delay
            }
    
    return results

def generate_ascii_chart(values, labels, title, max_width=50):
    """Generates an ASCII bar chart of the results"""
    
    if not values:
        return "No data to display"
    
    max_value = max(values)
    
    scale = max_width / max_value if max_value > 0 else 1
    
    chart = f"\n{title}\n" + "=" * len(title) + "\n\n"
    
    max_label_len = max(len(label) for label in labels)
    
    for i, (label, value) in enumerate(zip(labels, values)):
        bar_length = int(value * scale)
        bar = "#" * bar_length
        chart += f"{label.ljust(max_label_len)} | {value:.2f} | {bar}\n"
    
    return chart

def configure_tc_burst(switch, interface, burst_size):
    """
    Sets the burst capacity on the router (switch).
    """
    print(f"Configuring burst size {burst_size} packets on {switch.name} interface {interface}")
    
    current_qdisc = switch.cmd(f"tc qdisc show dev {interface}")
    
    switch.cmd(f"tc qdisc del dev {interface} root")
    
    # Configure HTB with specified burst capacity
    burst_bytes = burst_size * 1500
    
    cmd = f"tc qdisc add dev {interface} root handle 1: htb default 10"
    switch.cmd(cmd)
    
    # Add class with specified burst
    cmd = f"tc class add dev {interface} parent 1: classid 1:10 htb rate {DEFAULT_BW}Mbit burst {burst_bytes}"
    result = switch.cmd(cmd)
    
    # Verify config set correctly
    config = switch.cmd(f"tc class show dev {interface}")
    print(f"Burst configuration on {interface}:\n{config}")
    
    return result

def calculate_responsiveness_metrics(output_dir):
    """
    Calculate responsiveness (designed to simulate stall_rate).
    
    The calculation includes the following metrics:
    - Delay-throughput product: Goodput per unit of delay
    - Settling time: Time to return to steady state
    - Burst utilization: Utilization of burst capacity
    - Responsiveness index: Characterization of throughput relative to variance in delay
    """
    responsiveness_metrics = {}
    
    # Parse latency data
    for filename in os.listdir(output_dir):
        if filename.endswith('_ping_latency.log'):
            flow_name = filename.split('_')[0]
            if flow_name == 'h1':
                flow_name = 'TCP_Flow_1'
            elif flow_name == 'h2':
                flow_name = 'TCP_Flow_2'
                
            latency_file = os.path.join(output_dir, filename)
            
            iperf_file = os.path.join(output_dir, flow_name.lower().replace('tcp_flow_', 'h') + '_iperf_client.log')
            if not os.path.exists(iperf_file):
                continue
            
            # Parse ping data
            timestamps = []
            latencies = []
            
            with open(latency_file, 'r') as f:
                f.readline()
                
                for line in f:
                    if not line.strip():
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 2 or parts[1] == "NA":
                        continue
                    
                    try:
                        timestamps.append(float(parts[0]))
                        latencies.append(float(parts[1]))
                    except ValueError:
                        continue
            
            if not latencies:
                continue
                
            # Parse iperf goodput data
            iperf_timestamps = []
            throughputs = []
            
            with open(iperf_file, 'r') as f:
                for line in f:
                    if 'sec' in line and 'bits/sec' in line and not 'SUM' in line:
                        parts = line.split()
                        
                        # Extract timestamps
                        time_range = None
                        for part in parts:
                            if '-' in part and 'sec' in part:
                                time_range = part.split('-')
                                break
                        
                        if not time_range:
                            continue
                        
                        try:
                            timestamp = float(time_range[1].split('sec')[0])
                            
                            # Extract throughput
                            for i, part in enumerate(parts):
                                if 'bits/sec' in part and i > 0:
                                    value = float(parts[i-1])
                                    unit = part
                                    
                                    if 'Kbits/sec' in unit:
                                        value /= 1000
                                    elif 'Gbits/sec' in unit:
                                        value *= 1000
                                        
                                    iperf_timestamps.append(timestamp)
                                    throughputs.append(value)
                                    break
                        except (ValueError, IndexError):
                            continue
            
            if not throughputs:
                continue
                
            # Calculate metrics
            
            # Calculate delay variance (jitter)
            delay_variance = np.var(latencies) if len(latencies) > 1 else 0
            
            # Calculate throughput variance
            throughput_variance = np.var(throughputs) if len(throughputs) > 1 else 0
            
            # Calculate average throughput and latency
            avg_throughput = np.mean(throughputs)
            avg_latency = np.mean(latencies)
            
            # Calculate delay-throughput product
            delay_throughput_product = avg_latency / avg_throughput if avg_throughput > 0 else float('inf')
            
            # Calculate settling time (time when throughput reaches 90% of max steady throughput)
            # Use second half of the experiment as the "steady state"
            if len(throughputs) > 5:
                steady_state_throughput = np.mean(throughputs[len(throughputs)//2:])
                target_throughput = 0.9 * steady_state_throughput
                
                settling_time = None
                for i, tput in enumerate(throughputs):
                    if tput >= target_throughput:
                        settling_time = iperf_timestamps[i]
                        break
            else:
                settling_time = None
            
            # Calculate burst utilization
            peak_throughput = max(throughputs) if throughputs else 0
            burst_utilization = (avg_throughput / peak_throughput * 100) if peak_throughput > 0 else 0
            
            # Calculate responsiveness index
            # Combined metric to characterize throughput together with delay variances
            # Formula: (throughput / (1 + log(1 + delay_variance)))
            if delay_variance > 0:
                responsiveness_index = avg_throughput / (1 + np.log1p(delay_variance))
            else:
                responsiveness_index = avg_throughput
            
            # Store metrics
            responsiveness_metrics[flow_name] = {
                'responsiveness_index': responsiveness_index,
                'delay_throughput_product': delay_throughput_product,
                'burst_utilization': burst_utilization
            }
            
            if settling_time is not None:
                responsiveness_metrics[flow_name]['settling_time'] = settling_time
    
    return responsiveness_metrics

def bbrs_experiment(bw=DEFAULT_BW, delay=DEFAULT_DELAY, 
                   queue_size=DEFAULT_QUEUE_SIZE, duration=DEFAULT_DURATION,
                   output_dir=DEFAULT_OUTPUT_DIR, buffer_size=DEFAULT_BUFFER_SIZE, 
                   burst_size=DEFAULT_BURST_SIZE, trial_num=1):
    """
    Sets up experimental conditions and runs BBRS experiment.
    """
    trial_dir = os.path.join(output_dir, f"trial_{trial_num}")
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
    
    # Save config
    config_file = os.path.join(trial_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write(f"Experiment Configuration:\n")
        f.write(f"Bandwidth: {bw} Mbps\n")
        f.write(f"RTT: {delay*2} ms\n")
        f.write(f"Router Buffer Size: {buffer_size} packets ({buffer_size*1500/1000000:.2f} MB)\n")
        f.write(f"Router Burst Capacity: {burst_size} packets ({burst_size*1500/1000000:.2f} MB)\n")
        f.write(f"Duration: {duration} seconds\n")
        f.write(f"Trial: {trial_num}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configure topology
    topo = DumbbellTopo(
        bw=bw, 
        delay=delay, 
        queue_size=queue_size, 
        buffer_size=buffer_size, 
        burst_size=burst_size
    )
    
    # Create the network with OVSSwitch, CPU limited hosts, TCLink, and given topology
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink, switch=OVSSwitch)
    
    # Start the network
    net.start()
    print(f"*** Network started - Trial {trial_num}")
    
    # Get host objects
    sender1 = net.get('h1')
    sender2 = net.get('h2')
    receiver1 = net.get('h3')
    receiver2 = net.get('h4')
    s1 = net.get('s1')
    s2 = net.get('s2')
    
    # Configure BBR-S on all hosts
    for host in [sender1, sender2]:
        print(f"Setting up BBRS congestion control on {host.name}")
        host.cmd('sysctl -w net.ipv4.tcp_congestion_control=bbrs')
    
    if trial_num == 1:
        print("*** Dumping network connections")
        dumpNodeConnections(net.hosts)
    
    # Find the interface for the bottleneck link on s1
    s1_interfaces = s1.cmd('ip -o link show | grep -v "lo:" | cut -d: -f2').strip().split('\n')
    bottleneck_interface = None
    if s1_interfaces:
        # Find the interface that connects to s2
        for iface in s1_interfaces:
            iface = iface.strip()
            if iface and 's2' in s1.cmd(f"ip -o link show {iface}"):
                bottleneck_interface = iface
                break
        
        if not bottleneck_interface and s1_interfaces:
            bottleneck_interface = s1_interfaces[-1].strip()
    
    if bottleneck_interface:
        print(f"*** Configuring burst capacity on bottleneck interface {bottleneck_interface}")
        # Configure burst capacity on bottleneck
        configure_tc_burst(s1, bottleneck_interface, burst_size)
    else:
        print("*** WARNING: Could not identify bottleneck interface for burst configuration")
    
    # Start iperf servers on receivers
    server1_log = os.path.join(trial_dir, 'h3_iperf_server.log')
    server2_log = os.path.join(trial_dir, 'h4_iperf_server.log')
    
    start_iperf_server(receiver1, server1_log)
    start_iperf_server(receiver2, server2_log)
    
    # Wait for servers to start
    time.sleep(2)
    
    # Start TCP statistics monitoring
    sender1_stats = os.path.join(trial_dir, 'h1_tcp_stats.log')
    sender2_stats = os.path.join(trial_dir, 'h2_tcp_stats.log')
    
    monitor1 = Process(target=monitor_tcp_stats, args=(sender1, sender1_stats, duration))
    monitor2 = Process(target=monitor_tcp_stats, args=(sender2, sender2_stats, duration))
    
    monitor1.start()
    monitor2.start()
    
    # Monitor queue size on the bottleneck
    queue_stats_file = os.path.join(trial_dir, 'bottleneck_queue_stats.log')
    queue_monitor = None
    
    if bottleneck_interface:
        queue_monitor = Process(target=monitor_queue_size, 
                              args=(s1, bottleneck_interface, queue_stats_file, duration))
        queue_monitor.start()
    
    # Start ping latency monitoring
    ping1_log = os.path.join(trial_dir, 'h1_ping_latency.log')
    ping2_log = os.path.join(trial_dir, 'h2_ping_latency.log')
    
    ping1_monitor = Process(target=measure_ping_latency, 
                          args=(sender1, receiver1.IP(), ping1_log, duration))
    ping2_monitor = Process(target=measure_ping_latency, 
                          args=(sender2, receiver2.IP(), ping2_log, duration))
    
    ping1_monitor.start()
    ping2_monitor.start()
    
    # Start iperf clients
    client1_log = os.path.join(trial_dir, 'h1_iperf_client.log')
    client2_log = os.path.join(trial_dir, 'h2_iperf_client.log')
    
    start_iperf_client(sender1, receiver1.IP(), duration, client1_log, "bbrs")
    start_iperf_client(sender2, receiver2.IP(), duration, client2_log, "bbrs")
    
    print(f"*** Experiment running for {duration} seconds...")
    
    time.sleep(duration + 5)  # Add a small buffer between runs
    
    # Terminate experiment
    monitor1.terminate()
    monitor2.terminate()
    ping1_monitor.terminate()
    ping2_monitor.terminate()
    
    if queue_monitor:
        queue_monitor.terminate()
    
    # Stop the network
    net.stop()
    
    print(f"*** Trial {trial_num} completed. Results saved to {trial_dir}")
    
    # Calculate metrics for this trial
    metrics = calculate_buffer_bloat_metrics(trial_dir)
    responsiveness = calculate_responsiveness_metrics(trial_dir)
    
    all_metrics = {}
    all_metrics.update(metrics)
    
    for flow, resp_metrics in responsiveness.items():
        if flow in all_metrics:
            all_metrics[flow].update(resp_metrics)
        else:
            all_metrics[flow] = resp_metrics
    
    # Write to metrics file and return
    metrics_file = os.path.join(trial_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics

def run_multiple_trials(trials=DEFAULT_TRIALS, output_dir=DEFAULT_OUTPUT_DIR, **kwargs):
    """
    Runs multiple trials of the experiment.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_trial_metrics = []
    
    # Run trials
    for trial in range(1, trials + 1):
        print(f"\n\n*** Starting Trial {trial}/{trials} ***")
        
        trial_metrics = bbrs_experiment(output_dir=output_dir, trial_num=trial, **kwargs)
        all_trial_metrics.append(trial_metrics)
        
        # Wait between trials
        if trial < trials:
            print(f"Waiting 5 seconds before starting next trial...")
            time.sleep(5)
    
    # Calculate aggregate statistics
    aggregate_metrics = aggregate_trial_metrics(all_trial_metrics)
    
    # Save metrics
    agg_metrics_file = os.path.join(output_dir, 'aggregate_metrics.json')
    with open(agg_metrics_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    generate_summary_report(aggregate_metrics, all_trial_metrics, output_dir)
    
    return aggregate_metrics

def aggregate_trial_metrics(all_trial_metrics):
    """
    Calculates aggregate statistics across all trials.
    """
    if not all_trial_metrics:
        return {}
    
    aggregate = {}
    
    metric_values = {}
    
    for trial_metrics in all_trial_metrics:
        for flow, metrics in trial_metrics.items():
            if flow not in metric_values:
                metric_values[flow] = {}
            
            for metric_name, value in metrics.items():
                if metric_name not in metric_values[flow]:
                    metric_values[flow][metric_name] = []
                
                metric_values[flow][metric_name].append(value)
    
    # Calculate stats for each metric
    for flow, metrics in metric_values.items():
        aggregate[flow] = {}
        
        for metric_name, values in metrics.items():
            if not values:
                continue
                
            numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            
            if not numeric_values:
                continue
            
            # Calculate stats
            aggregate[flow][f"{metric_name}_mean"] = float(mean(numeric_values))
            aggregate[flow][f"{metric_name}_median"] = float(median(numeric_values))
            
            if len(numeric_values) > 1:
                aggregate[flow][f"{metric_name}_stdev"] = float(stdev(numeric_values))
            else:
                aggregate[flow][f"{metric_name}_stdev"] = 0.0
    
    return aggregate

def generate_summary_report(aggregate_metrics, all_trial_metrics, output_dir):
    """
    Generates a summary report.
    """
    report_file = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("===================================================\n")
        f.write("BBRS PERFORMANCE EVALUATION SUMMARY REPORT\n")
        f.write("===================================================\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of trials: {len(all_trial_metrics)}\n\n")
        
        # Write config
        config_file = os.path.join(output_dir, 'trial_1', 'config.txt')
        if os.path.exists(config_file):
            with open(config_file, 'r') as cf:
                f.write("EXPERIMENT CONFIGURATION\n")
                f.write("-----------------------\n")
                f.write(cf.read())
                f.write("\n\n")
        
        # Write key metrics summary
        f.write("KEY METRICS SUMMARY (Mean ± Stdev)\n")
        f.write("--------------------------------\n\n")
        
        key_metrics = [
            ("Queue Statistics", "queue_stats", ["avg_queue_size_mean", "max_queue_delay_mean", "p95_queue_delay_mean"]),
            ("TCP Flow 1", "h1", ["goodput_mean", "avg_rtt_mean", "inflation_ratio_mean", "responsiveness_index_mean"]),
            ("TCP Flow 2", "h2", ["goodput_mean", "avg_rtt_mean", "inflation_ratio_mean", "responsiveness_index_mean"])
        ]
        
        for section_name, flow_key, metrics in key_metrics:
            if flow_key == 'h1':
                actual_keys = [k for k in aggregate_metrics.keys() if 'h1' in k.lower() or 'flow_1' in k.lower()]
                flow_key = actual_keys[0] if actual_keys else flow_key
            elif flow_key == 'h2':
                actual_keys = [k for k in aggregate_metrics.keys() if 'h2' in k.lower() or 'flow_2' in k.lower()]
                flow_key = actual_keys[0] if actual_keys else flow_key
            
            if flow_key in aggregate_metrics:
                f.write(f"{section_name}:\n")
                
                flow_metrics = aggregate_metrics[flow_key]
                for metric in metrics:
                    if metric in flow_metrics:
                        mean_val = flow_metrics[metric]
                        stdev_metric = metric.replace('_mean', '_stdev')
                        stdev_val = flow_metrics.get(stdev_metric, 0)
                        
                        display_name = metric.replace('_mean', '').replace('_', ' ').title()
                        
                        f.write(f"  - {display_name}: {mean_val:.2f} ± {stdev_val:.2f}\n")
                
                f.write("\n")
    
        # Generate ASCII charts for metrics
        f.write("\nPERFORMANCE VISUALIZATIONS\n")
        f.write("-------------------------\n\n")
        
        # Comparison of goodput for different flow
        if all('goodput_mean' in aggregate_metrics.get(flow, {}) for flow in ['TCP_Flow_1', 'TCP_Flow_2']):
            goodput_values = [
                aggregate_metrics['TCP_Flow_1']['goodput_mean'],
                aggregate_metrics['TCP_Flow_2']['goodput_mean']
            ]
            goodput_labels = ['Flow 1', 'Flow 2']
            
            goodput_chart = generate_ascii_chart(
                goodput_values, 
                goodput_labels, 
                "Average Goodput (Mbps)"
            )
            f.write(goodput_chart + "\n\n")
        
        # Queue delay graph
        if 'queue_stats' in aggregate_metrics and 'avg_queue_delay_mean' in aggregate_metrics['queue_stats']:
            queue_values = [
                aggregate_metrics['queue_stats']['avg_queue_delay_mean'],
                aggregate_metrics['queue_stats'].get('p95_queue_delay_mean', 0),
                aggregate_metrics['queue_stats'].get('max_queue_delay_mean', 0)
            ]
            queue_labels = ['Average', '95th Percentile', 'Maximum']
            
            queue_chart = generate_ascii_chart(
                queue_values, 
                queue_labels, 
                "Queue Delay (ms)"
            )
            f.write(queue_chart + "\n\n")
        
        # Throughput and efficiency graph
        if all('efficiency_mean' in aggregate_metrics.get(flow, {}) for flow in ['TCP_Flow_1', 'TCP_Flow_2']):
            efficiency_values = [
                aggregate_metrics['TCP_Flow_1']['efficiency_mean'],
                aggregate_metrics['TCP_Flow_2']['efficiency_mean']
            ]
            efficiency_labels = ['Flow 1', 'Flow 2']
            
            efficiency_chart = generate_ascii_chart(
                efficiency_values, 
                efficiency_labels, 
                "Throughput/Delay Efficiency"
            )
            f.write(efficiency_chart + "\n\n")
        
        # Table of stats
        f.write("\nDETAILED STATISTICS\n")
        f.write("------------------\n\n")
        
        for flow, metrics in aggregate_metrics.items():
            f.write(f"{flow}:\n")
            
            sorted_metrics = sorted(metrics.items())
            
            for metric, value in sorted_metrics:
                f.write(f"  - {metric}: {value:.4f}\n")
            
            f.write("\n")
    
    print(f"Summary report generated: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='BBRS Algorithm Testing with Mininet')
    
    parser.add_argument('--bw', type=int, default=DEFAULT_BW,
                        help=f'Bandwidth in Mbps (default: {DEFAULT_BW})')
    parser.add_argument('--delay', type=int, default=DEFAULT_DELAY,
                        help=f'One-way delay in ms (default: {DEFAULT_DELAY})')
    parser.add_argument('--queue', type=int, default=DEFAULT_QUEUE_SIZE,
                        help=f'Queue size in packets (default: {DEFAULT_QUEUE_SIZE})')
    parser.add_argument('--buffer', type=int, default=DEFAULT_BUFFER_SIZE,
                        help=f'Router buffer size in packets (default: {DEFAULT_BUFFER_SIZE})')
    parser.add_argument('--burst', type=int, default=DEFAULT_BURST_SIZE,
                        help=f'Router burst capacity in packets (default: {DEFAULT_BURST_SIZE})')
    parser.add_argument('--duration', type=int, default=DEFAULT_DURATION,
                        help=f'Experiment duration in seconds (default: {DEFAULT_DURATION})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'Number of trials to run (default: {DEFAULT_TRIALS})')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode (open Mininet CLI)')
    
    args = parser.parse_args()
    
    setLogLevel('info')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output}_{timestamp}"
    
    print(f"Starting BBRS evaluation with {args.trials} trials...")
    print(f"Results will be saved to: {output_dir}")
    
    # Run trials
    aggregate_metrics = run_multiple_trials(
        trials=args.trials,
        output_dir=output_dir,
        bw=args.bw,
        delay=args.delay,
        queue_size=args.queue,
        buffer_size=args.buffer,
        burst_size=args.burst,
        duration=args.duration
    )
    
    print("\nAggregate Metrics Summary:")
    print("-------------------------")
    
    # Print aggregate metrics
    try:
        # Queue stats
        if 'queue_stats' in aggregate_metrics:
            queue_metrics = aggregate_metrics['queue_stats']
            print(f"Average Queue Size: {queue_metrics.get('avg_queue_size_mean', 'N/A'):.2f} packets")
            print(f"Average Queue Delay: {queue_metrics.get('avg_queue_delay_mean', 'N/A'):.2f} ms")
        
        # Flow 1 metrics
        flow1_key = next((k for k in aggregate_metrics.keys() if 'h1' in k.lower() or 'flow_1' in k.lower()), None)
        if flow1_key:
            flow1 = aggregate_metrics[flow1_key]
            print(f"\nFlow 1 Goodput: {flow1.get('goodput_mean', 'N/A'):.2f} Mbps")
            print(f"Flow 1 RTT: {flow1.get('avg_rtt_mean', 'N/A'):.2f} ms")
        
        # Flow 2 metrics
        flow2_key = next((k for k in aggregate_metrics.keys() if 'h2' in k.lower() or 'flow_2' in k.lower()), None)
        if flow2_key:
            flow2 = aggregate_metrics[flow2_key]
            print(f"\nFlow 2 Goodput: {flow2.get('goodput_mean', 'N/A'):.2f} Mbps")
            print(f"Flow 2 RTT: {flow2.get('avg_rtt_mean', 'N/A'):.2f} ms")
    except Exception as e:
        print(f"Error displaying metrics summary: {e}")
    
    print(f"\nDetailed results and report available in {output_dir}")
    print("Experiment completed successfully!")

if __name__ == '__main__':
    main()