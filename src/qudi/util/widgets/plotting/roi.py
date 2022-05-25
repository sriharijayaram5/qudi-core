# -*- coding: utf-8 -*-

"""
Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-core/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

__all__ = ['RectangleROI']

from math import isinf
from typing import Union, Tuple, Optional, Sequence, List
from PySide2 import QtCore
from pyqtgraph import ROI


class RectangleROI(ROI):
    """
    """
    def __init__(self,
                 pos=(0, 0),
                 size=(1, 1),
                 bounds=None,
                 apply_bounds_to_center=False,
                 parent=None,
                 pen=None,
                 hoverPen=None,
                 handlePen=None,
                 handleHoverPen=None,
                 movable=True,
                 resizable=True,
                 aspectLocked=False
                 ) -> None:
        ROI.__init__(self,
                     (0, 0),
                     size=(1, 1),
                     angle=0,
                     invertible=True,
                     maxBounds=None,
                     scaleSnap=False,
                     translateSnap=False,
                     rotateSnap=False,
                     parent=parent,
                     pen=pen,
                     hoverPen=hoverPen,
                     handlePen=handlePen,
                     handleHoverPen=handleHoverPen,
                     movable=movable,
                     rotatable=False,
                     resizable=resizable,
                     removable=False,
                     aspectLocked=aspectLocked)
        self.__center_position = (0, 0)
        self.__norm_size = (1, 1)
        self._apply_bounds_to_center = bool(apply_bounds_to_center)
        self._bounds = self.normalize_bounds(bounds)
        self.set_area(position=pos, size=size)

    @property
    def area(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.__center_position, self.__norm_size

    def set_area(self,
                 position: Optional[Tuple[float, float]] = None,
                 size: Optional[Tuple[float, float]] = None
                 ) -> None:
        if position is not None:
            self.setPos(QtCore.QPointF(position[0] - self.__norm_size[0] / 2,
                                       position[1] + self.__norm_size[1] / 2),
                        update=False,
                        finish=False)
            self.__center_position = (position[0], position[1])
        if size is not None:
            size = (abs(size[0]), abs(size[1]))
            self.setSize(QtCore.QPointF(size[0], -size[1]),
                         center=(0.5, 0.5),
                         update=False,
                         finish=False)
            self.__norm_size = size
        if position is not None and size is not None:
            self._clip_area(update=True, finish=True)

    @property
    def bounds(self) -> List[Tuple[Union[None, float], Union[None, float]]]:
        return self._bounds.copy()

    def set_bounds(self,
                   bounds: Union[None, Sequence[Tuple[Union[None, float], Union[None, float]]]]
                   ) -> None:
        self._bounds = self.normalize_bounds(bounds)
        self._clip_area(update=True, finish=True)

    def _clip_area(self, update: Optional[bool] = True, finish: Optional[bool] = True) -> None:
        position = list(self.__center_position)
        size = list(self.__norm_size)
        x_min, x_max = self._bounds[0]
        y_min, y_max = self._bounds[1]
        if self._apply_bounds_to_center:
            if (x_min is not None) and (position[0] < x_min):
                position[0] = x_min
            elif (x_max is not None) and (position[0] > x_max):
                position[0] = x_max
            if (y_min is not None) and (position[1] < y_min):
                position[1] = y_min
            elif (y_max is not None) and (position[1] > y_max):
                position[1] = y_max
        else:
            left = position[0] - size[0] / 2
            right = position[0] + size[0] / 2
            top = position[1] + size[1] / 2
            bottom = position[1] - size[1] / 2
            if (x_min is not None) and (left < x_min):
                position[0] = x_min + size[0] / 2
            elif (x_max is not None) and (right > x_max):
                position[0] = x_max - size[0] / 2
            if (y_min is not None) and (bottom < y_min):
                position[1] = y_min + size[1] / 2
            elif (y_max is not None) and (top > y_max):
                position[1] = y_max - size[1] / 2
        rect_pos = self.pos()
        current_pos = QtCore.QRectF(rect_pos, self.size()).center()
        translate = (position[0] - current_pos.x(), position[1] - current_pos.y())
        self.__center_position = tuple(position)
        self.setPos(rect_pos.x() + translate[0],
                    rect_pos.y() + translate[1],
                    update=update,
                    finish=finish)

    @staticmethod
    def normalize_bounds(bounds: Union[None, Sequence[Tuple[Union[None, float], Union[None, float]]]]
                         ) -> List[Tuple[Union[None, float], Union[None, float]]]:
        if bounds is None:
            bounds = [(None, None), (None, None)]
        else:
            bounds = [list(span) for span in bounds]
            # Replace inf values by None
            for span in bounds:
                for ii, val in enumerate(span):
                    try:
                        if isinf(val):
                            span[ii] = None
                    except TypeError:
                        pass
            # Sort spans in ascending order
            try:
                bounds[0] = tuple(sorted(bounds[0]))
            except TypeError:
                bounds[0] = tuple(bounds[0])
            try:
                bounds[1] = tuple(sorted(bounds[1]))
            except TypeError:
                bounds[1] = tuple(bounds[1])
        return bounds

    # @staticmethod
    # def normalize_rect(pos: Tuple[float, float], size: Tuple[float, float]) -> QtCore.QRectF:
    #     try:
    #         pos = QtCore.QPointF(pos[0], pos[1])
    #     except TypeError:
    #         pass
    #     try:
    #         size = QtCore.QSizeF(size[0], size[1])
    #     except TypeError:
    #         pass
    #     x_min, x_max = sorted([pos.x(), pos.x() + size.width()])
    #     y_min, y_max = sorted([pos.y(), pos.y() + size.height()])
    #     return QtCore.QRectF(x_min,
    #                          y_max,
    #                          abs(size.width()),
    #                          -abs(size.height()))

    def checkPointMove(self, handle, pos, modifiers):
        pos = self.mapSceneToParent(pos)
        x_min, x_max = self._bounds[0]
        y_min, y_max = self._bounds[1]
        if (x_min is not None) and pos.x() < x_min:
            return False
        if (x_max is not None) and pos.x() > x_max:
            return False
        if (y_min is not None) and pos.y() < y_min:
            return False
        if (y_max is not None) and pos.y() > y_max:
            return False
        return True

    def mouseDragEvent(self, ev) -> None:
        if not ev.isAccepted():
            if self.translatable and ev.button() == QtCore.Qt.LeftButton and ev.modifiers() == QtCore.Qt.NoModifier:
                is_start = ev.isStart()
                is_finish = ev.isFinish()
                ev.accept()
                if is_start:
                    self.setSelected(True)
                    self._moveStarted()
                if self.isMoving:
                    translate = self.mapToParent(ev.pos()) - self.mapToParent(ev.buttonDownPos())
                    new_pos = self.preMoveState['pos'] + translate
                    self.__center_position = (self.__center_position[0] + translate.x(),
                                              self.__center_position[1] + translate.y())
                    self.setPos(new_pos, update=False, finish=False)
                    self._clip_area(update=True, finish=False)
                if is_finish:
                    self._moveFinished()
            else:
                ev.ignore()