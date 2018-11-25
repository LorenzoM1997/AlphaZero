from tkinter import *
from tkinter.font import Font
from time import sleep

class slot(Label):
    def __init__(self, mainFrame):

        super().__init__(mainFrame,
                         font=displayFont,
                         height=1,
                         width=4,)

# construct UI
def Checker_Init(rows, cols, array):

    # display
    global root, displayFont
    global simulationsArray
    simulationsArray= array
    root = Tk()
    root.configure(background='black')
    displayFont = Font(family='Helvetica',
                       size=65,
                       weight='bold')

    # initialize frames
    topFrame = Frame(root)
    topFrame.pack()
    mainFrame = Frame(root)
    mainFrame.pack()
    bottomFrame = Frame(root)
    bottomFrame.pack()

    # specify labels
    title = Label(topFrame,
                  text='Checker Simulations',
                  font=('Elephant', 30),
                  fg='purple',
                  bg='black')
    title.pack()

    bottom_label = Label(bottomFrame,
                         text='Moves count:',
                         font=('Elephant', 20),
                         fg='purple',
                         bg='black')
    bottom_label.pack()

    # build game board grid
    slots = []
    slots = generate_grid(rows, cols, mainFrame)

    # build buttons
    global buttons, pauseCounter
    buttons = []
    pauseCounter = 0

    # start button + move counter
    buttons.append(Button(root,
                          text='Start', font=('Helvetica', 15, 'bold'),
                          height=1,
                          width=4,
                          padx=10,
                          pady=10,
                          bg='azure3',
                          command=ExecuteSimulations))
    # pause button 
    buttons.append(Button(root,
                          text='Pause',
                          font=('Helvetica', 15, 'bold'),
                          height=1,
                          width=4,
                          padx=10,
                          pady=10,
                          bg='azure3',
                          command=Pause))

    buttons[0].pack()  
    buttons[1].pack() 
    root.mainloop()


# run through simulations when triggered
def ExecuteSimulations():
    moveCount = 0
    num_cols = len(simulationsArray[0][0])

    for i in range(0, len(simulationsArray)):  # tic-tac-toe states
        for r in range(0, len(simulationsArray[0])):  # rows
            for c in range(0, len(simulationsArray[0][0])):  # cols
                if (simulationsArray[i][r][c] == 1):
                    slots[r*num_cols + c].config(text='O',
                                                 font=displayFont,
                                                 fg='red',
                                                 bg='white')
                elif (simulationsArray[i][r][c] == 2):
                    slots[r*num_cols + c].config(text='O',
                                                 font=displayFont,
                                                 fg='blue',
                                                 bg='white')
                else:
                    slots[r*num_cols + c].config(text=' ',
                                                 font=displayFont,
                                                 bg='white')
        moveCount += 1
        buttons[0].config(text=str(moveCount))

        root.update()
        sleep(3)


# needs multi-threading, crashes when trying to pause mid-execution
def Pause():
    global pauseCounter
    pauseCounter += 1
    while True:
        if (pauseCounter % 2 == 0):
            break  # when clicked twice
        else:
            sleep(2)


def generate_grid(rows, cols, mainFrame):
    global slots
    slots = []
    for r in range(rows):
        for c in range(cols):
            new_slot = slot(mainFrame)
            slots.append(new_slot)
            new_slot.grid(row=r,
                          column=c,
                          padx=10,
                          pady=10)
            new_slot.config(bg='white')
    return slots
