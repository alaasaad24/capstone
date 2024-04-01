import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sample data for demonstration
portfolio_value = [100, 110, 115, 120, 118, 122, 130]
initial_cash = 100

fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], label='Portfolio Value', color='blue')
initial_cash_line = ax.axhline(y=initial_cash, color='r', linestyle='--', label='Initial Cash')
ax.set_xlabel('Time')
ax.set_ylabel('Portfolio Value')
ax.set_title('Stock Trading Simulation')
ax.legend()

def init():
    ax.set_xlim(0, len(portfolio_value))
    ax.set_ylim(min(portfolio_value) * 0.9, max(portfolio_value) * 1.1)
    return line, initial_cash_line

def update(frame):
    x_data = list(range(1, frame + 1))
    y_data = portfolio_value[:frame]
    line.set_data(x_data, y_data)
    if frame == len(portfolio_value):
        ani.event_source.stop()  # Stop animation when last point is reached
    return line, initial_cash_line

ani = FuncAnimation(fig, update, frames=len(portfolio_value) + 1, init_func=init, blit=True, interval=200)
plt.show()
