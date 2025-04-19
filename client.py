import customtkinter as ctk
import time
import requests
import numpy as np

class HomePage(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid(row=0, column=0, sticky="nsew")
        self.master = master

        self.create_widgets()

    def pack(self, **kwargs):
        # self.master.geometry(f"{self.winfo_width()}x{self.winfo_height()}")
        self.master.update_idletasks()
        self.create_widgets()
        super().pack(fill="both", expand=True, **kwargs)
        self.pack_widgets()

    def create_widgets(self):
        self.title_label = ctk.CTkLabel(self, text="T5 Chat", font=("Helvetica", 35))
        self.subtitle_label = ctk.CTkLabel(self, text="Welcome to T5 Chat!", text_color=('#808080', '#e6e6e6'), font=("Helvetica", 15))
        self.signup_button = ctk.CTkButton(self, text="Sign Up", font=("Helvetica", 20), command=self.signup, width=170, height=40)
        self.login_button = ctk.CTkButton(self, text="Log In", font=("Helvetica", 20), command=self.login, width=170, height=40)

    def pack_widgets(self):
        self.title_label.pack(pady=(10, 0), anchor=ctk.N)
        self.subtitle_label.pack(pady=(2, 100), anchor=ctk.N)
        self.signup_button.pack(pady=0, anchor=ctk.CENTER)
        self.login_button.pack(pady=40, anchor=ctk.CENTER)

    def signup(self):
        self.master.hsignup()

    def login(self):
        self.master.hlogin()

    def destroy(self):
        super().destroy()
        self.title_label.destroy()

class SignupPage(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid(row=1, column=0, sticky="nsew")
        self.master = master

    def create_widgets(self):
        self.back_button = ctk.CTkButton(self, height=40, width=170, text="Back", font=("Helvetica", 20), command=self.master.back)
        self.title_label = ctk.CTkLabel(self, text="T5 Chat", font=("Helvetica", 35))
        self.subtitle_label = ctk.CTkLabel(self, text="Sign up", text_color=('#808080', '#e6e6e6'), font=("Helvetica", 18))
        self.username_entry = ctk.CTkEntry(self, font=("Helvetica", 15), placeholder_text='Username', width=250)
        self.password_entry = ctk.CTkEntry(self, font=("Helvetica", 15), placeholder_text='Password', width=250, show='•')
        self.signup_button = ctk.CTkButton(self, text="Sign Up", font=("Helvetica", 20), command=self.master.signup, width=170, height=40)

    def pack(self):
        super().pack(fill="both", expand=True)
        self.create_widgets()
        self.back_button.pack(pady=(0, 150), side = ctk.BOTTOM)
        self.title_label.pack(pady=(10, 0), anchor=ctk.N)
        self.subtitle_label.pack(pady=(2, 10), anchor=ctk.N)
        self.username_entry.pack(pady=(30, 0), anchor=ctk.CENTER)
        self.password_entry.pack(pady=(10, 0), anchor=ctk.CENTER)
        self.signup_button.pack(pady=(25, 0), anchor=ctk.CENTER)

    def destroy(self):
        super().destroy()
        self.back_button.destroy()
        self.title_label.destroy()
        self.subtitle_label.destroy()
        self.username_entry.destroy()
        self.password_entry.destroy()

class LoginPage(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid(row=1, column=0, sticky="nsew")
        self.master = master

    def create_widgets(self):
        self.back_button = ctk.CTkButton(self, height=40, width=170, text="Back", font=("Helvetica", 20), command=self.master.back)
        self.title_label = ctk.CTkLabel(self, text="T5 Chat", font=("Helvetica", 35))
        self.subtitle_label = ctk.CTkLabel(self, text="Log In", text_color=('#808080', '#e6e6e6'), font=("Helvetica", 18))
        self.username_entry = ctk.CTkEntry(self, font=("Helvetica", 15), placeholder_text='Username', width=250)
        self.password_entry = ctk.CTkEntry(self, font=("Helvetica", 15), placeholder_text='Password', width=250, show='•')
        self.signup_button = ctk.CTkButton(self, text="Log In", font=("Helvetica", 20), command=self.master.login, width=170, height=40)

    def pack(self):
        super().pack(fill="both", expand=True)
        self.create_widgets()
        self.back_button.pack(pady=(0, 150), side = ctk.BOTTOM)
        self.title_label.pack(pady=(10, 0), anchor=ctk.N)
        self.subtitle_label.pack(pady=(2, 10), anchor=ctk.N)
        self.username_entry.pack(pady=(30, 0), anchor=ctk.CENTER)
        self.password_entry.pack(pady=(10, 0), anchor=ctk.CENTER)
        self.signup_button.pack(pady=(25, 0), anchor=ctk.CENTER)

    def destroy(self):
        super().destroy()
        self.back_button.destroy()
        self.title_label.destroy()
        self.subtitle_label.destroy()
        self.username_entry.destroy()
        self.password_entry.destroy()

class ConvPage(ctk.CTkFrame):
    ...

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("T5 Chat")
        self.geometry("700x500")

        self.home_page = HomePage(self, width=700, height=500)
        self.home_page.pack()

        self.username = None
        self.password = None

    @property
    def logged_in(self):
        return (bool(self.username) and bool(self.password))

    def hsignup(self):
        self.home_page.destroy()
        self.signup_page = SignupPage(self, width=700, height=500)
        self.signup_page.pack()

    def hlogin(self):
        self.home_page.destroy()
        self.signup_page = LoginPage(self, width=700, height=500)
        self.signup_page.pack()

    def signup(self):
        global IP
        username = self.signup_page.username_entry.get()
        password = self.signup_page.password_entry.get()
        url = f"http://{IP}/api/signup"
        data = {"username": str(username), "password": str(password)}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            if response.json()["success"] == False:
                print(response.json()["message"])
                return
            
            self.signup_page.destroy()
            self.home_page = ConvPage(self, width=1500, height=750)
            self.geometry('1500x750')
            self.home_page.pack()
            self.username = username
            self.password = password

    def login(self): 
        global IP
        username = self.signup_page.username_entry.get()
        password = self.signup_page.password_entry.get()
        url = f"http://{IP}/api/login"
        data = {"username": str(username), "password": str(password)}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            if response.json()["success"] == False:
                print(response.json()["message"])
                return

            self.signup_page.destroy()
            self.home_page = ConvPage(self, width=1500, height=750)
            self.geometry('1500x750')

    def back(self):
        self.signup_page.destroy()
        self.home_page = HomePage(self, width=700, height=500)
        self.home_page.pack()

ctk.set_appearance_mode("dark")

IP = input("Please input the server's IP address: ")

app = App()
app.mainloop()
