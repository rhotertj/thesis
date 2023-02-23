class LabelDecoder:

    def __init__(self, num_classes: int) -> None:
        """Infers class names and their integer mapping
        from the number of classes. 

        Args:
            num_classes (int): Number of classes.
        """
        self.num_classes = num_classes
        self.class_names = self.get_classnames()
        self.decode_event = self.choose_label_mapping()

    def __call__(self, x: dict) -> int:
        """Maps the input to an integer.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        return self.decode_event(x)

    def get_classnames(self) -> list[str]:
        """Returns the classnames based on the number of classes.

        Returns:
            list[str]: Class names.
        """
        if self.num_classes == 2:
            return ["Background", "Action"]

        cls_names = ["Background", "Pass", "Shot", "Foul"]
        return cls_names[:self.num_classes]

    def choose_label_mapping(self) -> callable:
        """Choose correct function to decode event annotations based on the number of classes.

        Raises:
            ValueError: Unknown number of classes.

        Returns:
            callable: Function that maps annotations to integers.
        """
        if self.num_classes == 2:
            return self.has_action
        if self.num_classes == 3:
            return self.background_pass_shot
        if self.num_classes == 4:
            return self.background_pass_shot_foul
        else:
            raise ValueError(f"Number of classes ({self.num_classes}) is invalid!")

    def has_action(self, x: dict):
        """Decodes actionness.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        if x == {}:
            return 0
        return 1

    def background_pass_shot_foul(self, x):
        """Decodes pass, shots and foul.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        if x == {}:
            return 0
        if x["Pass"] == "O" and x["Wurf"] == "0":
            return 3
        return self.background_pass_shot(x)

    def background_pass_shot(self, x):
        """Decodes passes and shots.

        Args:
            x (dict): The event annotation.

        Returns:
            int: Integer label.
        """
        if x == {}:
            return 0
        if not x["Pass"] in ("O", "X"):
            pass_labels = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
            return pass_labels[x["Pass"]]
        elif not x["Wurf"] == "0":
            shot_labels = {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2, "7": 2, "8": 2}
            return shot_labels[x["Wurf"]]
        return 0
